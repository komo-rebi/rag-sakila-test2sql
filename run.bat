@echo off
echo ========================================
echo RAG Text2SQL 评估系统
echo ========================================
echo.

echo 正在启动评估系统...
cd src

echo 运行Text2SQL评估（模拟模式）...
python run_evaluation.py --config ../config/salila_config_real_mock.yaml

echo.
echo 评估完成！
echo 请查看 results/ 目录下的结果文件
echo - HTML报告: evaluation_report_*.html
echo - JSON结果: evaluation_results_*.json
echo.

pause