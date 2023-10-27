import argparse, os
import lanes

def parse_args():
    parser = argparse.ArgumentParser(description="CVIP Project")
    parser.add_argument(
        "--input_folder", type=str, default='images',
        help="Path to Input Directory")
    parser.add_argument(
        "--output_folder", type=str, default="output",
        help="Path to the Output Folder")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    input_dir = args.input_folder
    output_dir = args.output_folder
    os.makedirs(output_dir, exist_ok=True)
    
    images = list()

    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            images.append(file)
    
    for img in images:

        img_path = os.path.join(input_dir,img)
        output_image_matrix = lanes.find_lane(img_path)

        output_image_path = os.path.join(output_dir,img)
        lanes.save_image(output_image_matrix, output_image_path)

if __name__ == "__main__":
    main()