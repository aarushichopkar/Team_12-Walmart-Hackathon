from flask import Flask, render_template, request
import pandas as pd
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


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save('test.csv')
        
        # Load the uploaded CSV file into a DataFrame
        df_test = pd.read_csv('test.csv')
        
        # Perform fraud detection using your model
        processed_df = process_uploaded_file('test.csv')
        
    
        
        # Save updated flagged_ips to pickle file
        with open('metrics_flagged_ips.pkl', 'wb') as f:
            pickle.dump({'metrics': metrics, 'flagged_ips': flagged_ips}, f)

    return redirect('/')  # Redirect back to the main page

if __name__ == '__main__':
    app.run(debug=True)

    