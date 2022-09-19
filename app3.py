
import codecs 
import os
from flask import Flask, request, render_template,url_for, redirect, Blueprint,send_file
import app1 as app1
from googletrans import Translator
from docx import Document
from flask import send_from_directory

app_file3 = Flask(__name__) 
app_file3 = Blueprint('app_file3',__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
translator = Translator()

def transhindi():
    doc=Document()
    with open('uploads/tes_op.txt', 'r', encoding='utf-8') as file:
        with open("uploads/hindi.txt", "w",) as f1:
            contents=file.read()
            result=translator.translate(contents, dest='en')
            result1=result.text
            f1.write(result1)
    return('hindi.txt')

def transenglish():
    
#contents = text.read()
    doc=Document()
    with open('uploads/tes_op.txt', 'r', encoding='utf-8') as file:
        with open("uploads/english.txt", "w",) as f1:
            contents=file.read()
            result=translator.translate(contents, dest='en')
            result1=result.text
            f1.write(result1)
    return('english.txt')
 







@app_file3.route("/translate")
def translate():
    return render_template('translate.html')

@app_file3.route("/filetranslate",methods=['GET','POST'])
def return_files_tut():
    try:   
        if request.method == "POST":
         req = request.form
         userText= req.get("language")
        #userText = request.form['pdf']
        #userText1 = request.form['docx']
         print(str(userText))
         #print(str(userText1))
         if(userText=="hindi"):
            res=transhindi()
            return redirect('/downloadtranslate/'+ res)
            return render_template('translate.html')
         elif(userText=="english"):
            res=transenglish()
            return redirect('/downloadtranslate/'+ res)
            return render_template('translate.html')
         else:
            return render_template('translate.html')
        return redirect(request.url)
    except Exception as e:
        return str(e)
    return None    




@app_file3.route("/downloadtranslate/<res>", methods = ['GET'])
def download_translate(res):
    return render_template('downloads2.html',value=res)

@app_file3.route('/return-transfiles/<res>')
def return_transfiles(res):
    return send_file('uploads/'+res, as_attachment=True, attachment_filename='')
    return render_template('home.html')












