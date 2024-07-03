import ffmpeg
import os

def convert_dav_to_mp4(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file)
            .run()
        )
        print(f"Successfully converted {input_file} to {output_file}")
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf8') if e.stderr else "No stderr available"
        print(f"An error occurred while converting {input_file}: {stderr_output}")

def convert_files_in_directory(input_directory, output_directory, output_format='mp4'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.dav'):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.{output_format}")
            print(f"Converting {input_file} to {output_file}...")
            convert_dav_to_mp4(input_file, output_file)

if __name__ == "__main__":
    input_directory = "video_06_30"  # Specify the path to the folder with .dav files
    output_directory = "converted_video_06_30"  # Specify the path to the folder for saving output files
    if not os.path.exists(input_directory):
        print(f"Input directory {input_directory} does not exist. Please check the path.")
    else:
        convert_files_in_directory(input_directory, output_directory)
