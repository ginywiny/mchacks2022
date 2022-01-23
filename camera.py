from email.mime import image
import cv2
import os
import image_difference

def main(show_pics=True):

    camera = cv2.VideoCapture("/dev/video2")
    workdir = os.path.dirname(__file__)
    images_dir = os.path.join(workdir, "images")
    before_image = None

    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    while True:
        check, frame = camera.read()
        if before_image is None:
            before_image = frame

        if show_pics:
            cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        elif key == 32:
            # Space key pressed

            # name = input("Object name: ")
            # object_dir = os.path.join(images_dir, name)

            # if not os.path.isdir(object_dir):
            #     os.mkdir(object_dir)

            # num_files = len(os.listdir(object_dir))
            # print(f"num files: {num_files}")
            # cv2.imwrite(os.path.join(object_dir, f"{num_files:04d}.png"), frame)

            try:
                image_diff = image_difference.calculate_image_difference(before_image, frame)

                bbox = image_difference.predict_removed_object(before_image, frame)
                cropped_image = image_difference.crop_bbox_from_image(bbox, before_image)

                if show_pics:
                    cv2.imshow("image_diff", image_diff)
                    cv2.imshow("cropped_image", cropped_image)
            
            except Exception as e:
                print(f"could not compute image difference: {e}")

            before_image = frame
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()