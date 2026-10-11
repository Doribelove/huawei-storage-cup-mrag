@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%PROJECT_ROOT%\..") do set "TASK_ROOT=%%~fI"
set "PYTHONPATH=%PROJECT_ROOT%\src"
set "PYTHONIOENCODING=utf-8"
cd /d "%PROJECT_ROOT%"
echo Full-source workbench: http://127.0.0.1:8766
echo Requires the verified local data, completed index and fixed model files described in README.
echo Keep this console open. Ctrl+C stops. Qwen has not run; candidates are not answers.
"%TASK_ROOT%\bge-dense-lab\.venv\Scripts\python.exe" -u -m evidence_lab.workbench --audit-dir "%TASK_ROOT%\multimodal-evidence-lab\output\audit" --phase-dir "%TASK_ROOT%\multimodal-evidence-lab\output\audit\phase2" --freeze-dir "%PROJECT_ROOT%\output\runs\full-v1-20261005" --model-dir "%TASK_ROOT%\bge-dense-lab\models\bge-m3" --reranker-dir "%PROJECT_ROOT%\models\bge-reranker-v2-m3" --port 8766
if errorlevel 1 pause
endlocal
