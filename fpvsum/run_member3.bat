@echo off
REM Member 3 single-video inference: run_member3.bat <video.mp4> <out.json>
pushd "%~dp0"
python -m member3_fpvsum.infer %1 %2
popd
