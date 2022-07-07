#import required libraries
from curses.ascii import US
import streamlit as st
import numpy as np 
import pandas as pd 
# import sklearn

from PIL import Image
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

string = "Classification Of Students Academic Performance"

im = Image.open("Student_Performance_Chart.jpg")
im1 = Image.open("Student_AIM.png")

male_img = Image.open("MaleStudentCartoonjpg.jpg")
female_img = Image.open("FemaleStudentCartoon.jpg")

egypt_img = Image.open("egypt.jpg")
iran_img = Image.open("iran.jpg")
iraq_img = Image.open("iraq.jpg")
USA_img = Image.open("USA.jpg")
jordan_img = Image.open("jordan.jpg")
KW_img = Image.open("KW.jpg")
lebanon_img = Image.open("lebanon.jpg")
Lybia_img = Image.open("Lybia.jpg")
Morocco_img = Image.open("Morocco.jpg")
Palestine_img = Image.open("Palestine.jpg")
SaudiArabia_img = Image.open("SaudiArabia.jpg")
Syria_img = Image.open("Syria.jpg")
Tunis_img = Image.open("Tunis.jpg")
venzuela_img = Image.open("venzuela.jpg")







st.set_page_config(page_title=string, page_icon='chart_with_upwards_trend',layout="wide",
              initial_sidebar_state="auto", menu_items=None)

st.title (string, anchor=None)





student = pd.read_csv("xAPI-Edu-Data.csv")
#remove the features which are not usefull
student.drop(['GradeID','SectionID','Topic'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

categ = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'Semester',
       'Relation', 'raisedhands', 'VisITedResources', 'AnnouncementsView',
       'Discussion', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
       'StudentAbsenceDays', 'Class']

student[categ] = student[categ].apply(le.fit_transform)



y = student['Class'] # Class is the value we want to predict

X = student[['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'Semester',
       'Relation', 'raisedhands', 'VisITedResources', 'AnnouncementsView',
       'Discussion', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
       'StudentAbsenceDays']]



# from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)


def shuffle_split_data(X, y):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 70)

    X_train = X[split]
    X_test =  X[~split]
    y_train = y[split]
    y_test = y[~split]

#     print len(X_Train), len(y_Train), len(X_Test), len(y_Test)
    return X_train, X_test, y_train, y_test

X_train,X_test,y_train,y_test = shuffle_split_data(X, y)

#-------------------------------------------------------------


#-------------------------------------------------------------



rfc = RandomForestClassifier(n_estimators=80,max_features='auto', max_depth=9,min_samples_leaf=1,
                             min_samples_split=2,bootstrap=True, random_state = 42)

rfc.fit(X_train,y_train)
rfcpred = rfc.predict(X_test)

# print("Random Forest Classifier Accuracy Accuracy: ")
# Tunned_forest = accuracy_score(rfcpred,y_test) 

# confusion_matrix(y_test, rfcpred)



with st.sidebar:
       st.image(im1)
       st.markdown("""---""")
       st.title('About this Webapp')
       st.write('As the students performance is very important for any organisation or institute.'
       'Primarily this is webapp aims to determine the category of students in which they are lying and'
       ' gives intuitive feed back for their improvement. This helps to detect early'
       'raise and fall of students performance.')

       st.markdown("""---""")
       
       st.markdown("**Connect wtih me 😊** [LINK](https://www.linkedin.com/in/vedant-mukhedkar-4864881b0/)")
       st.markdown("**My Github** [LINK](https://www.github.com/git5v)")
       
       st.markdown("""---""")



st.image(im,  width = 600)
st.markdown("""---""")
st.markdown('**Please enter the below fields correctly**')

genre1 = st.radio(
     ' Please select the gender of student ',
     ('Female', 'Male'))

if genre1 == 'Female':
     st.image(female_img,  width = 250)
else:
     st.image(male_img,  width = 250)

st.markdown("""---""")

genre2 = st.radio(
     'Please select the Nationality of student ',
       ('Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco', 
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon', 'venzuela'
       ))

if genre2 == 'Egypt':
     st.image(egypt_img,  width = 300)
elif(genre2 == 'Iran'):
     st.image(iran_img,  width = 300)
elif(genre2 == 'Iraq'):
     st.image(iraq_img,  width = 300)
elif(genre2 == 'Jordan'):
     st.image(jordan_img,  width = 300)
elif(genre2 == 'KW'):
     st.image(KW_img,  width = 300)
elif(genre2 == 'Lybia'):
     st.image(Lybia_img,  width = 300)
elif(genre2 == 'Morocco'):
     st.image(Morocco_img,  width = 300)
elif(genre2 == 'Palestine'):
     st.image(Palestine_img,  width = 300)
elif(genre2 == 'SaudiArabia'):
     st.image(SaudiArabia_img,  width = 300)
elif(genre2 == 'Syria'):
     st.image(Syria_img,  width = 300)    
elif(genre2 == 'Tunis'):
     st.image(Tunis_img,  width = 300)
elif(genre2 == 'USA'):
     st.image(USA_img,  width = 300)
elif(genre2 == 'lebanon'):
     st.image(lebanon_img,  width = 300)
elif(genre2 == 'venzuela'):
     st.image(venzuela_img,  width = 300)

st.markdown("""---""")


genre3 = st.radio(
     'Please select the Place of birth of student ',
       ('Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco', 
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon', 'venzuela'
       ))

if genre3 == 'Egypt':
     st.image(egypt_img,  width = 300)
elif(genre3 == 'Iran'):
     st.image(iran_img,  width = 300)
elif(genre3 == 'Iraq'):
     st.image(iraq_img,  width = 300)
elif(genre3 == 'Jordan'):
     st.image(jordan_img,  width = 300)
elif(genre3 == 'KW'):
     st.image(KW_img,  width = 300)
elif(genre3 == 'Lybia'):
     st.image(Lybia_img,  width = 300)
elif(genre3 == 'Morocco'):
     st.image(Morocco_img,  width = 300)
elif(genre3 == 'Palestine'):
     st.image(Palestine_img,  width = 300)
elif(genre3 == 'SaudiArabia'):
     st.image(SaudiArabia_img,  width = 300)
elif(genre3 == 'Syria'):
     st.image(Syria_img,  width = 300)    
elif(genre3 == 'Tunis'):
     st.image(Tunis_img,  width = 300)
elif(genre3 == 'USA'):
     st.image(USA_img,  width = 300)
elif(genre3 == 'lebanon'):
     st.image(lebanon_img,  width = 300)
elif(genre3 == 'venzuela'):
     st.image(venzuela_img,  width = 300)
st.markdown("""---""")

genre4 = st.radio(
     'Please select the Stage of student ',
       ('High School', 'Middle School', 'Lower level'
       ))

if genre4 == 'High School':
     st.markdown('**Student is  in higher school**')
elif(genre4 == 'Middle School'):
     st.markdown('**Student is  in Middle school**')
else:
       st.markdown('**Student is  in Lower level school**')

st.markdown("""---""")
       
genre5 = st.radio(
     'Please select the Semester of student ',
       ('Fall', 'Spring'
       ))

if genre5 == 'Fall':
     st.markdown('**Fall Semester**')
else:
       st.markdown('**Spring Semester**')
st.markdown("""---""")

genre6 = st.radio(
     'Relation of Guardian',
       ('Father', 'Mother'
       ))

if genre6 == 'Father':
     st.markdown('**Father**')
else:
       st.markdown('**Mother**')
st.markdown("""---""")


number1 = st.number_input('Enter the total raise hands of student from 0 to 100')
if(number1>100 or number1<0): st.write("Please enter the valid value")
else : st.write('The current number is ', number1) 
st.markdown("""---""")

number2 = st.number_input('Enter how many times student visited resources from 0 to 100')
if(number2>100 or number2<0): st.write("Please enter the valid value")
else : st.write('The current number is ', number2) 
st.markdown("""---""")

number3 = st.number_input('Enter how many times the announcement is viewed by student from 0 to 100')
if(number3>100 or number3<0): st.write("Please enter the valid value")
else : st.write('The current number is ', number3) 
st.markdown("""---""")

number4 = st.number_input('Enter how many times student taken part in discussion from 0 to 100')
if(number4>100 or number4<0): st.write("Please enter the valid value")
else : st.write('The current number is ', number4) 
st.markdown("""---""")


option1 = st.selectbox(
     'Parent Answering Survey',
     ('No','Yes'))
st.write('You selected:', option1)
st.markdown("""---""")

option2 = st.selectbox(
     'Parent School Satisfaction',
     ('Bad','Good'))
st.write('You selected:', option2)
st.markdown("""---""")

option3 = st.selectbox(
     'Student Absent Days ',
     ('Above-7','Below-7'))
st.write('You selected:', option3)
st.markdown("""---""")

Dict1 = {"Female": 0, "Male": 1}
Dict2 = {'Egypt': 0, 'Iran': 1, 'Iraq': 2, 'Jordan': 3, 'KW': 4, 'Lybia': 5, 'Morocco': 6, 'Palestine': 7, 'SaudiArabia': 8, 'Syria': 9, 'Tunis': 10, 'USA': 11, 'lebanon': 12, 'venzuela': 13}
Dict3 = {'High School': 0, 'Middle School': 1, 'Lower level': 2}
Dict4 = {'Fall': 0, 'Spring': 1}
Dict5 = {"Father": 0, "Mother": 1}
Dict6 = {"No": 0, "Yes": 1}
Dict7 = {"Bad": 0, "Good": 1}
Dict8 = {"Below-7": 1, "Above-7": 0}



genre1 = Dict1[genre1]
genre2 = Dict2[genre2]
genre3 = Dict2[genre3]
genre4 = Dict3[genre4]
genre5 = Dict4[genre5]
genre6 = Dict5[genre6]
option1 = Dict6[option1]
option2 = Dict7[option2]
option3 = Dict8[option3]




X_new = np.array([[genre1,genre2,genre3,genre4,genre5,genre6,number1,number2,number3
                     ,number4,option1,option2,option3]])

#Prediction of the species from the input vector
prediction = rfc.predict(X_new)
# st.markdown("Prediction of Species: {}".format(prediction))

if prediction[0]==0:
       st.balloons()
       st.success("Congratulations! The student is catagorised in HIGH level")
elif prediction[0]==1:
       st.success("Unfortunately the student is in LOW level catagory")
else: st.success("The student is in MEDIUM level catagory")



if st.button('Click if you wants to know hte analysis of studemnts'):
       st.title('Here the analysis of student')

       #Bar Chart
       st.bar_chart(student['raisedhands'][50:100:])
       st.write('Avarage Raised Hands of students',student['raisedhands'].mean())

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 0:
                     cnt += 1
                     val += student['raisedhands'][xx]
              xx += 1
       handRaiseOfHighStd = val/cnt
       st.write('Avarage raise hands of HIGH level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 1:
                     cnt += 1
                     val += student['raisedhands'][xx]
              xx += 1
       handRaiseOfLowStd = val/cnt
       st.write('Avarage raise hands of LOW level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 2:
                     cnt += 1
                     val += student['raisedhands'][xx]
              xx += 1
       handRaiseOfMedStd = val/cnt 
       st.write('Avarage raise hands of MEDIIUM level students',val/cnt)
       
       if(number1<handRaiseOfHighStd):
              st.title("Feedback")
              st.markdown('The student is not asking questions whenever your are having doubt by rasing hands' 
              'So **try to raise hands in class and get doubts clear right there**')
       else: 
          st.title("Feedback")
          st.markdown('**You are doing good here just continue what you are doing**')

       st.markdown("""---""")

       st.bar_chart(student['VisITedResources'][:50:])
       st.write('Avarage VisITed Resources of students',student['VisITedResources'].mean())

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 0:
                     cnt += 1
                     val += student['VisITedResources'][xx]
              xx += 1
       visRecHigh = val/cnt 
       st.write('Avarage Visited Rescources of HIGH level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 1:
                     cnt += 1
                     val += student['VisITedResources'][xx]
              xx += 1
       visRecLow = val/cnt 
       st.write('Avarage Visited Rescources of LOW level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 2:
                     cnt += 1
                     val += student['VisITedResources'][xx]
              xx += 1
       visRecMed = val/cnt 
       st.write('Avarage Visited Rescources of MEDIUM level students',val/cnt)

       if(number2<visRecHigh):
              st.title("Feedback")
              st.markdown('The student is not visiting the resources frequently ' 
              'So **try to utilize the resources and visit them frequently whenever you get a chance**')
       else: 
          st.title("Feedback")
          st.markdown('**You are doing good here just continue what you are doing**')
          
       st.markdown("""---""")
       
       st.bar_chart(student['Discussion'][:50:])
       st.write('Avarage Discussion of students',student['Discussion'].mean())

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 0:
                     cnt += 1
                     val += student['Discussion'][xx]
              xx += 1
       disHigh = val/cnt 
       st.write('Avarage Discussion of HIGH level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 1:
                     cnt += 1
                     val += student['Discussion'][xx]
              xx += 1
       disLow = val/cnt 
       st.write('Avarage Discussion of LOW level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 2:
                     cnt += 1
                     val += student['Discussion'][xx]
              xx += 1
       disMed = val/cnt 
       st.write('Avarage Discussion of MEDIUM level students',val/cnt)

       if(number3<disHigh):
              st.title("Feedback")
              st.markdown('It seems that the student take very less paricipation in discussions' 
              'So **try to take participate more and more in discussion to get to know various interesting things**')
       else: 
          st.title("Feedback")
          st.markdown('**You are doing good here just continue what you are doing**')

       st.markdown("""---""")

       st.bar_chart(student['AnnouncementsView'][:50:])
       st.write('Avarage Announcements View of students',student['AnnouncementsView'].mean())

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 0:
                     cnt += 1
                     val += student['AnnouncementsView'][xx]
              xx += 1
       viewH = val/cnt 
       st.write('Avarage AnnouncementsView of HIGH level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 1:
                     cnt += 1
                     val += student['AnnouncementsView'][xx]
              xx += 1
       viewL = val/cnt 
       st.write('Avarage Announcements View of LOW level students',val/cnt)

       val = 0
       cnt = 0
       xx = 0
       for i in student['Class']:
              if i == 2:
                     cnt += 1
                     val += student['AnnouncementsView'][xx]
              xx += 1
       viewL = val/cnt 
       st.write('Avarage Announcements View of MEDIUM level students',val/cnt)

       if(number4<viewH):
              st.title("Feedback")
              st.markdown('It seems that the student is not much concerned about announcement that happens in universities' 
              'So **try to see the Announcements whenever get a chance so that you can not miss any oporituinites and updates**')
       else: 
          st.title("Feedback")
          st.markdown('**You are doing good here just continue what you are doing**')

       st.markdown("""---""")




footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style=' text-align: center;' 
href="https://www.linkedin.com/in/vedant-mukhedkar-4864881b0/v/" target="_blank">Vedant Mukhedkar</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
