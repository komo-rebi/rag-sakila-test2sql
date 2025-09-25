"""
Salila Text2SQL评估主程序
执行完整的Text2SQL系统评估流程
"""

import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

# 设置OpenAI API配置
import openai
openai.api_base = "https://api.gptsapi.net/v1"
openai.api_key = "{{APIKEY}}"

# 延迟导入Text2SQL系统，避免在模拟模式下产生不必要的警告
from evaluation_metrics import EvaluationMetrics, MetricsAggregator

class SalilaEvaluator:
    """Salila Text2SQL评估器"""
    
    def __init__(self, config_path: str):
        """
        初始化评估器
        
        Args:
            config_path (str): 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 创建输出目录（在setup_logging之前）
        self.output_dir = Path(self.config['evaluation']['output_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # 初始化组件
        # 检查是否启用模拟模式，如果是则直接使用简化版系统
        mock_mode = self.config.get('evaluation', {}).get('mock_mode', False)
        
        if mock_mode:
            self.logger.info("模拟模式：直接使用简化版Text2SQL系统")
            from text2sql_system_simple import SimpleText2SQLSystem
            self.text2sql_system = SimpleText2SQLSystem(self.config.get('model', {}))
        else:
            try:
                # 只有在非模拟模式下才导入完整版系统
                from text2sql_system import Text2SQLSystem
                self.text2sql_system = Text2SQLSystem(self.config.get('model', {}))
                self.logger.info("使用完整版Text2SQL系统")
            except Exception as e:
                self.logger.warning(f"完整版Text2SQL系统初始化失败: {e}")
                self.logger.info("回退到简化版Text2SQL系统")
                from text2sql_system_simple import SimpleText2SQLSystem
                self.text2sql_system = SimpleText2SQLSystem(self.config.get('model', {}))
        
        # 检查是否启用模拟模式
        mock_mode = self.config.get('evaluation', {}).get('mock_mode', False)
        self.metrics_calculator = EvaluationMetrics(
            self.config['database']['connection_string'],
            mock_mode=mock_mode
        )
        self.metrics_aggregator = MetricsAggregator(self.config)
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise Exception(f"配置文件加载失败: {e}")
    
    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config['evaluation'].get('log_level', 'INFO'))
        
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 文件处理器
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 配置根日志器
        logging.basicConfig(
            level=log_level,
            handlers=[console_handler, file_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"评估开始，配置文件: {self.config_path}")
    
    def load_test_dataset(self) -> List[Dict[str, Any]]:
        """加载测试数据集"""
        dataset_path = Path(self.config['evaluation']['dataset_path'])
        
        # 处理相对路径
        if not dataset_path.is_absolute():
            dataset_path = Path(self.config_path).parent / dataset_path
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            self.logger.info(f"成功加载测试数据集，共 {len(dataset)} 条记录")
            return dataset
            
        except Exception as e:
            self.logger.error(f"测试数据集加载失败: {e}")
            raise
    
    def evaluate_single_query(self, test_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个查询
        
        Args:
            test_item (dict): 测试项目
            
        Returns:
            dict: 评估结果
        """
        query_id = test_item.get('id', 'unknown')
        question = test_item['question']
        expected_sql = test_item.get('expected_sql', test_item.get('sql', ''))
        
        self.logger.info(f"评估查询 {query_id}: {question}")
        
        result = {
            'id': query_id,
            'question': question,
            'expected_sql': expected_sql,
            'generated_sql': None,
            'execution_accuracy': {},
            'syntax_accuracy': {},
            'semantic_similarity': {},
            'retrieval_recall': {},
            'error_message': None,
            'evaluation_time': None
        }
        
        start_time = datetime.now()
        
        try:
            # 1. 生成SQL
            generated_sql, retrieval_context = self.text2sql_system.generate_sql(
                question, return_context=True
            )
            result['generated_sql'] = generated_sql
            
            if not generated_sql:
                result['error_message'] = "SQL生成失败"
                return result
            
            # 2. 计算执行准确率
            if self.config['metrics']['execution_accuracy']['enabled']:
                result['execution_accuracy'] = self.metrics_calculator.calculate_execution_accuracy(
                    generated_sql, expected_sql
                )
            
            # 3. 计算语法正确率
            if self.config['metrics']['syntax_accuracy']['enabled']:
                result['syntax_accuracy'] = self.metrics_calculator.calculate_syntax_accuracy(
                    generated_sql
                )
            
            # 4. 计算语义相似度
            if self.config['metrics']['semantic_similarity']['enabled']:
                result['semantic_similarity'] = self.metrics_calculator.calculate_semantic_similarity(
                    generated_sql, expected_sql, question
                )
            
            # 5. 计算检索召回率
            if self.config['metrics']['retrieval_recall']['enabled']:
                retrieved_info = self.text2sql_system.get_retrieval_info(question)
                result['retrieval_recall'] = self.metrics_calculator.calculate_retrieval_recall(
                    retrieved_info, test_item
                )
            
        except Exception as e:
            self.logger.error(f"查询 {query_id} 评估失败: {e}")
            result['error_message'] = str(e)
        
        finally:
            result['evaluation_time'] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        运行完整评估
        
        Returns:
            dict: 评估结果
        """
        self.logger.info("开始Text2SQL系统评估")
        
        # 加载测试数据
        test_dataset = self.load_test_dataset()
        
        # 评估每个查询
        detailed_results = []
        for i, test_item in enumerate(test_dataset, 1):
            self.logger.info(f"进度: {i}/{len(test_dataset)}")
            
            result = self.evaluate_single_query(test_item)
            detailed_results.append(result)
            self.metrics_aggregator.add_result(result)
            
            # 保存中间结果（可选）
            if i % 5 == 0:  # 每5个查询保存一次
                self._save_intermediate_results(detailed_results, i)
        
        # 计算总体指标
        overall_metrics = self.metrics_aggregator.calculate_overall_metrics()
        
        # 生成最终报告
        final_report = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'config_file': str(self.config_path),
                'total_queries': len(test_dataset),
                'evaluation_duration': sum(r.get('evaluation_time', 0) for r in detailed_results)
            },
            'overall_metrics': overall_metrics,
            'detailed_results': detailed_results
        }
        
        # 保存结果
        self._save_results(final_report)
        
        # 生成报告
        if self.config['reporting']['generate_html']:
            self._generate_html_report(final_report)
        
        self.logger.info("评估完成")
        return final_report
    
    def _save_intermediate_results(self, results: List[Dict], count: int):
        """保存中间结果"""
        try:
            intermediate_file = self.output_dir / f"intermediate_results_{count}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"中间结果保存失败: {e}")
    
    def _save_results(self, report: Dict[str, Any]):
        """保存评估结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式
        if self.config['reporting']['generate_json']:
            json_file = self.output_dir / f"evaluation_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"结果已保存到: {json_file}")
        
        # 保存简化的指标摘要
        summary_file = self.output_dir / f"metrics_summary_{timestamp}.json"
        summary = {
            'timestamp': report['evaluation_info']['timestamp'],
            'total_queries': report['evaluation_info']['total_queries'],
            'overall_metrics': report['overall_metrics']
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """生成HTML报告"""
        try:
            html_content = self._create_html_content(report)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = self.output_dir / f"evaluation_report_{timestamp}.html"
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML报告已生成: {html_file}")
            
        except Exception as e:
            self.logger.error(f"HTML报告生成失败: {e}")
    
    def _create_html_content(self, report: Dict[str, Any]) -> str:
        """创建HTML报告内容"""
        overall = report['overall_metrics']
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text2SQL评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metric-title {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
                .metric-value {{ font-size: 24px; color: #007bff; }}
                .details {{ margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Text2SQL系统评估报告</h1>
                <p>评估时间: {report['evaluation_info']['timestamp']}</p>
                <p>总查询数: {report['evaluation_info']['total_queries']}</p>
                <p>评估耗时: {report['evaluation_info']['evaluation_duration']:.2f}秒</p>
            </div>
            
            <div class="metrics">
                {"" if "mock" in report['evaluation_info']['config_file'] else f'''
                <div class="metric-card">
                    <div class="metric-title">执行准确率</div>
                    <div class="metric-value">{overall['execution_accuracy']['mean']:.2%}</div>
                    <p>可执行率: {overall['execution_accuracy']['success_rate']:.2%}</p>
                </div>
                '''}
                <div class="metric-card">
                    <div class="metric-title">语法正确率</div>
                    <div class="metric-value">{overall['syntax_accuracy']['mean']:.2%}</div>
                    <p>有效率: {overall['syntax_accuracy']['valid_rate']:.2%}</p>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">语义相似度</div>
                    <div class="metric-value">{overall['semantic_similarity']['mean']:.2%}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">检索召回率</div>
                    <div class="metric-value">{overall['retrieval_recall']['overall_recall_mean']:.2%}</div>
                    <p>表级: {overall['retrieval_recall']['table_recall_mean']:.2%}</p>
                    <p>列级: {overall['retrieval_recall']['column_recall_mean']:.2%}</p>
                    <p>关键列: {overall['retrieval_recall']['key_column_recall_mean']:.2%}</p>
                    <p>权重: {overall['retrieval_recall']['weighted_recall_mean']:.2%}</p>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">综合得分</div>
                    <div class="metric-value">{overall['weighted_overall_score']:.2%}</div>
                </div>
            </div>
            
            <div class="details">
                <h2>召回率详细分析</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">表级分析</div>
                        <p>总期望表数: {overall['retrieval_recall']['table_analysis']['total_expected']}</p>
                        <p>总检索表数: {overall['retrieval_recall']['table_analysis']['total_retrieved']}</p>
                        <p>总匹配表数: {overall['retrieval_recall']['table_analysis']['total_matched']}</p>
                        <p>整体精确率: {overall['retrieval_recall']['table_analysis']['overall_precision']:.2%}</p>
                        <p>整体召回率: {overall['retrieval_recall']['table_analysis']['overall_recall']:.2%}</p>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">列级分析</div>
                        <p>总期望列数: {overall['retrieval_recall']['column_analysis']['total_expected']}</p>
                        <p>总检索列数: {overall['retrieval_recall']['column_analysis']['total_retrieved']}</p>
                        <p>总匹配列数: {overall['retrieval_recall']['column_analysis']['total_matched']}</p>
                        <p>整体精确率: {overall['retrieval_recall']['column_analysis']['overall_precision']:.2%}</p>
                        <p>整体召回率: {overall['retrieval_recall']['column_analysis']['overall_recall']:.2%}</p>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">关键列分析</div>
                        <p>总期望关键列数: {overall['retrieval_recall']['key_column_analysis']['total_expected']}</p>
                        <p>总匹配关键列数: {overall['retrieval_recall']['key_column_analysis']['total_matched']}</p>
                        <p>整体召回率: {overall['retrieval_recall']['key_column_analysis']['overall_recall']:.2%}</p>
                        <p>平均覆盖率: {overall['retrieval_recall']['key_column_analysis']['average_coverage']:.2%}</p>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">SQL类型分布</div>
                        {"".join([f"<p>{sql_type}: {count}条</p>" for sql_type, count in overall['retrieval_recall']['sql_type_analysis']['type_distribution'].items()])}
                    </div>
                </div>
                
                <h3>最常缺失的元素</h3>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">最常缺失的表</div>
                        {"".join([f"<p>{table}: {count}次</p>" for table, count in overall['retrieval_recall']['table_analysis']['most_missed_tables']])}
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">最常缺失的列</div>
                        {"".join([f"<p>{column}: {count}次</p>" for column, count in overall['retrieval_recall']['column_analysis']['most_missed_columns']])}
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">最常缺失的关键列</div>
                        {"".join([f"<p>{column}: {count}次</p>" for column, count in overall['retrieval_recall']['key_column_analysis']['most_missed_key_columns']])}
                    </div>
                </div>
                
                <h2>详细结果</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>问题</th>
                            {"" if "mock" in report['evaluation_info']['config_file'] else "<th>执行准确率</th>"}
                            <th>语法正确率</th>
                            <th>语义相似度</th>
                            <th>状态</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for result in report['detailed_results']:
            exec_acc = result.get('execution_accuracy', {}).get('execution_accuracy', 0)
            syntax_acc = result.get('syntax_accuracy', {}).get('syntax_accuracy', 0)
            semantic_sim = result.get('semantic_similarity', {}).get('semantic_similarity', 0)
            
            status = "成功" if not result.get('error_message') else "失败"
            status_class = "success" if status == "成功" else "error"
            
            exec_acc_cell = "" if "mock" in report['evaluation_info']['config_file'] else f"<td>{exec_acc:.2%}</td>"
            html += f"""
                        <tr>
                            <td>{result.get('id', 'N/A')}</td>
                            <td>{result.get('question', 'N/A')[:50]}...</td>
                            {exec_acc_cell}
                            <td>{syntax_acc:.2%}</td>
                            <td>{semantic_sim:.2%}</td>
                            <td class="{status_class}">{status}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Text2SQL系统Salila评估')
    parser.add_argument(
        '--config', 
        type=str, 
        default='../configs/salila_config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建评估器
        evaluator = SalilaEvaluator(args.config)
        
        # 运行评估
        results = evaluator.run_evaluation()
        
        # 打印摘要
        overall = results['overall_metrics']
        print("\n" + "="*50)
        print("评估完成！结果摘要:")
        print(f"总查询数: {overall['total_queries']}")
        
        # 检查是否启用执行准确率
        if results.get('evaluation_info', {}).get('config_file', '').find('mock') == -1:
            # 非模拟模式显示执行准确率
            print(f"执行准确率: {overall['execution_accuracy']['mean']:.2%}")
        
        print(f"语法正确率: {overall['syntax_accuracy']['mean']:.2%}")
        print(f"语义相似度: {overall['semantic_similarity']['mean']:.2%}")
        print(f"检索召回率: {overall['retrieval_recall']['overall_recall_mean']:.2%}")
        print(f"  - 表级召回率: {overall['retrieval_recall']['table_recall_mean']:.2%}")
        print(f"  - 列级召回率: {overall['retrieval_recall']['column_recall_mean']:.2%}")
        print(f"  - 关键列召回率: {overall['retrieval_recall']['key_column_recall_mean']:.2%}")
        print(f"  - 权重召回率: {overall['retrieval_recall']['weighted_recall_mean']:.2%}")
        print(f"综合得分: {overall['weighted_overall_score']:.2%}")
        print("="*50)
        
    except Exception as e:
        print(f"评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 