import subprocess

def call_evaluate(in_path, dspth):
    # venv_path =
    script_path = r"D:\Developed\face-parsing.PyTorch\face_mask_gen.py"
    command = [
        "Conda",
        "activate", 
        "bisenet_env",
        "&&",
        "Python",
        script_path,
        "--input_img",
        in_path,
        "--result_dir",
        dspth,

    ]
    print(" ".join(command))
    subprocess.run(command, shell=True)
    

def call_function():
    # Call the evaluate function from another virtual environment
    call_evaluate(r"C:\dumpinGGrounds\stuff_pg\outputs\Sherlyn\Sherlyn-Chopra-251.jpg",  r'C:\temp112')
call_function()