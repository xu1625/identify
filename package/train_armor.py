from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("run/train4/weights/last.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="data_sets/robot_armor.yaml",
                          epochs=400,
                          imgsz=640,
                          project='run'
                          )