from neuralnet import predictcnn


from loadataset import loading_image


def get_result(imagefile):
    X = loading_image(imagefile).flatten()
    predictions_cnn = predictcnn(X.reshape(-1,15,15,1))
    print(predictions_cnn)
    return predictions_cnn[0]

