# Davis Mechatronics Interview Challenge
## Challenge requirements
- input: pictures, output: x,y,z
- front and back is different, pictures are combined. can assume camera is fixed. 
- camera used is zed X
- possible that there's no pole too, e.g. 20250826_111428_front_frame000123_rgb, but not a primary concern
- to be run real time on nvidia orin

## Run instructions
1. create a virtualenv (optional but recommended): ``python -m venv .venv; .\.venv\Scripts\Activate.ps1``
2. install dependencies: ``pip install -r requirements.txt``
3. Run ``RUN-ME.py``: ``python RUN-ME.py INPUT-RGBD OUTPUT-JSON --draw`` or ``python RUN-ME.py INPUT-RGBD OUTPUT-JSON``
