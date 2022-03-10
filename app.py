from flask import Flask,jsonify,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

model = pickle.load(open("autism.pkl", "rb"))


@app.route('/')
def hello_world():
    #return render_template("index.html")
    return "hello world"

@app.route('/predict',methods=['POST'])
def predict():
    
    
    a1 = request.form.get('a1')
    a2 = request.form.get('a2')
    a3 = request.form.get('a3')
    a4 = request.form.get('a4')
    a5 = request.form.get('a5')
    a6 = request.form.get('a6')
    a7 = request.form.get('a7')
    a8 = request.form.get('a8')
    a9 = request.form.get('a9')
    a10 = request.form.get('a10')
    a11 = request.form.get('a11')
    a12 = request.form.get('a12')
    a13 = request.form.get('a13')
    a14 = request.form.get('a14')
    a15 = request.form.get('a15')
    input_query=np.array([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15]])
    #res={'a1':a1,'a2':a2,'a3':a3,'a4':a4,'a5':a5,'a6':a6,'a7':a7,'a8':a8,'a9':a9,'a10':a10,'a11':a11,'a12':a12,'a13':a13,'a14':a14,'a15':a15}
    #output =model.predict(input_query)[0]
    
    
    #row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15])])
    
    #print(row_df)
    #prediction=model.predict(input_query)[0]
    prediction=model.predict_proba(input_query)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output)*100)

    '''if output>str(0.5):
        return render_template('result.html',pred=f'You have chance of having ASD.\nProbability of having ASD is {output}')
    else:
        return render_template('result.html',pred=f'You are safe.\n Probability of having ASD is {output}')

    return jsonify(res)'''
    return jsonify({'output':str(output)})


if __name__ == '__main__':
    app.run(debug=True)
#git config --global user.email  sujithreddy.c.v@gmail
