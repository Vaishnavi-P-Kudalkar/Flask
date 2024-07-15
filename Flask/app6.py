# use fro loop and expression 

#integrate html with flask 
# http vern GET and POST method 

### buildinhgg url dynamically
###jinjang template 
''' 
{%.....%} for statements
{{    }}  expressions to print output
{#.....#} this is for comments
'''

from flask import Flask , redirect, url_for, render_template, request
#wsgi application - bw webserver and webapplication
app =Flask(__name__)

@app.route('/') #decorator, directs to the page(url in brackets) and this fn is executed
def welcome():
      return render_template('index.html') 


@app.route('/passed/<int:score>') #route should have a return statement 
def passed(score):
      ref=''
      if score>=50:
            res="PASS"
      else:
            res="FAIL"
      exp ={'score':score,'res':res}
      return render_template('result6.html',result=exp)

@app.route('/fail/<int:score>') #decorator, directs to the page(url in brackets) and this fn is executed
def fail(score):
      res='FAIL'

      return render_template('result.html',result=res)


@app.route('/submit/results/<int:marks>')
def resulted(marks):
      resulted = ""
      if marks<50:
            resulted='fail'
      else:
            resulted= "passed"
      return redirect(url_for(resulted,score=marks))#redirected to passed or fail page 

#result checker html page
@app.route('/submit',methods=['POST','GET'])
def submit():
      total_score=0
      if request.method =='POST':
            science=float(request.form['science'])
            math=float(request.form['math'])
            datascience=float(request.form['datascience'])
            total_score = (science + math + datascience)/3
      res=''
      if total_score>=50:
            res='passed'
      else:
            res='fail'
      return redirect(url_for(res, score=total_score))



if __name__ =='__main__':
    app.run(debug=True)# debug= makes it partly real time 