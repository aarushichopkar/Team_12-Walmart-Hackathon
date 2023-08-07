from flask import Flask, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    # Load metrics and flagged IPs from pickle file
    with open('metrics_flagged_ips.pkl', 'rb') as f:
        data = pickle.load(f)
        metrics = data['metrics']
        flagged_ips = data['flagged_ips']

    return render_template('index.html', metrics=metrics, flagged_ips=flagged_ips)

if __name__ == '__main__':
    app.run(debug=True)
    