from flask import Flask, render_template, request,Blueprint
from app1 import app_file1
from app2 import app_file2
from app3 import app_file3
main_app = Flask(__name__)
main_app.register_blueprint(app_file1)
main_app.register_blueprint(app_file2)
main_app.register_blueprint(app_file3)
@main_app.route("/")
def indexmain():
    return render_template("home.html")   

if __name__=='__main__':
   	main_app.run(debug=True, use_reloader= False)
       