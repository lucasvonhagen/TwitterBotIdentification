from flask import render_template
from app import app
from markupsafe import escape


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title="Index")


