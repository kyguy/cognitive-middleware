'''
FILE: aggregate_data.py
DESC: Traverses file tree searching for files of TYPE and
      appends them into a single file for processing
'''
import os

PATH="./application-templates"
TYPE="json"
OUT ="templates.txt"

def extract():
 for cur, _dirs, files in os.walk(PATH):
   pref = ''
   head, tail = os.path.split(cur)
   while head:
     head, _tail = os.path.split(head)
     for f in files:
       sep = str(f).split(".")
       if(len(sep) > 1):
         if(sep[1] == TYPE):
           if("secret" not in f):
             print(f)
             with open(cur + "/" + f, 'r') as f2:
               f1.write(f2.read())

with open(OUT, 'w') as f1:
  extract()
  f1.close()

