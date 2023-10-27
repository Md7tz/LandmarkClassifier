import glob
import subprocess
import datetime
import tarfile


def create_submit_pkg():
    # Source files
    src_files = glob.glob("src/*.py")

    # Notebooks
    notebooks = glob.glob("*.ipynb")

    # Generate HTML files from the notebooks

    for nb in notebooks:
        cmd_line = f"jupyter nbconvert --to html {nb}"

        print(f"Executing {cmd_line}")
        subprocess.check_call(cmd_line, shell=True)

    html_files = glob.glob("*.htm*")

    now = (
        datetime.datetime.today().isoformat(timespec="minutes").replace(":", "h") + "m"
    )
    outfile = f"submission_{now}.tar.gz"

    print(f"Adding files to {outfile}")

    with tarfile.open(outfile, "w:gz") as tar:
        for name in src_files + notebooks + html_files:
            print(name)
            tar.add(name)

    print("")
    msg = f"Done. Please upload {outfile} to the Udacity workspace"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))


if __name__ == "__main__":
    create_submit_pkg()