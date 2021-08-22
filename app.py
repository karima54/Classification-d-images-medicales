import os
from flask import Flask, render_template, request, redirect
from inference import get_prediction
from commons import format_class_name, create_predictions_folders
from commons import get_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    predicted_files = []
    # check model
    get_model()
    if request.method == 'POST':
        # create the folders to stock the predictions in
        predictions_dir = create_predictions_folders()
        if 'file' not in request.files:
            print("redirection")
            return redirect(request.url)
        files = request.files.getlist("file")
        for file in files:
            print(file.filename)
            if not file:
                return
            img_bytes = file.read()
            class_name, class_id = get_prediction(image_bytes=img_bytes)
            # predicted_files.append({"file_name": file.filename, "class_id": class_id, "class_name": class_name})
            if class_name in os.listdir(predictions_dir):
                print(class_name)
                # app.config['UPLOAD_FOLDER'] = os.path.join(predictions_dir, class_name, file.filename)
                file_path = os.path.join(predictions_dir, class_name, file.filename)
                file.stream.seek(0)
                file.save(file_path)
            predicted_files.append({"file_name": file.filename, "class_name": class_name, "file_path": file_path})
        print("predicted_files", predicted_files)
        return render_template('result.html', predicted_files=predicted_files)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
