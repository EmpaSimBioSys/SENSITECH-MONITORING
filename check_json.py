from flask import Flask, request, render_template_string, jsonify
import json

app = Flask(__name__)

index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>JSON Viewer</title>
</head>
<body>
    <h1>JSON Viewer</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    {% if json_data %}
        <div>
            <h2>JSON Data</h2>
            <ul>
                {% for key in json_data.keys() %}
                    <li><a href="#" onclick="fetchData('{{ key }}')">{{ key }}</a></li>
                {% endfor %}
            </ul>
        </div>
        <div id="content">
            <h3>Content will be shown here</h3>
        </div>
    {% endif %}
    <script>
        function fetchData(key) {
            fetch(`/data/${key}`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById("content").innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                });
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(index_html)

@app.route('/upload', methods=['POST'])
def upload_file():
    global json_data
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.json'):
        json_data = json.load(file)
        return render_template_string(index_html, json_data=json_data)
    return 'Invalid file type'

@app.route('/data/<key>', methods=['GET'])
def get_data(key):
    global json_data
    if key in json_data:
        return jsonify(json_data[key])
    return jsonify({})

if __name__ == '__main__':
    json_data = {}
    app.run(debug=True, port=8000)
