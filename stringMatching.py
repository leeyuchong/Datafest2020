import pandas as pd
from fuzzywuzzy import process, fuzz
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from urllib.parse import urlparse
import time

def googleSearch(query):
    g_clean = [] #this is the list we store the search results
    url = 'https://www.google.com/search?client=ubuntu&channel=fs&q={}&ie=utf-8&oe=utf-8'.format(query)#//this is the actual query we are going to scrape
    try:
            html = requests.get(url)
            #print(html.text)
            soup = BeautifulSoup(html.text, 'lxml')
            a = soup.find_all('a')# // a is a list
            for i in a:
                k = i.get('href')
                try:
                    m = re.search(r"(?P<url>https?://[^\s]+)", k)
                    n = m.group(0)
                    rul = n.split('&')[0]
                    domain = urlparse(rul)
                    if(re.search('google.com', domain.netloc)):
                        continue
                    else:
                        g_clean.append(rul)
                        break
                except:
                    continue
    except Exception as ex:
        print(str(ex))
    return g_clean

check = pd.read_csv("College-Plans_updatedNames4.csv")
checkNames = check["institution"]
against = pd.read_csv("CollegeFinances_editedCopy.csv")
againstNames = against["Name"]

fullNameList = []
i=0

for e in againstNames:
    fullNameList.append(e)
'''
# Extract differences
for name in checkNames:
    #print(name)
    if name not in fullNameList:
        print(name)
        i+=1
        # print(str(i)+": "+name)
        # if name.startswith("('"):
        #     end = name.index("',")
        #     newName = name[2:end]
        #     check_updated.at[i, 'institution'] = newName
    #i+=1   
#check_updated.to_csv('College-Plans_updatedNames3.csv')
print(i)
'''

# check duplicates
nameList = []
for e in checkNames:
    nameList.append(e)
for e in checkNames:
    if nameList.count(e)>=2:
        print(str(i)+": "+e)
    i+=1
print(i)

'''
# Google Search Matching
for name in checkNames:
    if name not in fullNameList:
        match = process.extractOne(name,fullNameList)
        matchResult = googleSearch(match[0])
        print(matchResult[0])
        nameResult = googleSearch(name)
        print(nameResult[0])
        if matchResult[0] == nameResult[0]:
            print("i: "+str(i)+" "+name+" => "+match[0])
            check.at[i, 'institution'] = match[0]
        else:
            match = process.extractOne(name,fullNameList, scorer = fuzz.token_set_ratio)
            matchResult = googleSearch(match[0])
            print(matchResult[0])
            print(nameResult[0])
            if matchResult[0] == nameResult[0]:
                print("i: "+str(i)+" "+name+" => "+match[0])
                check.at[i, 'institution'] = match[0]
        time.sleep(4)
        # print("i: "+str(i)+" "+name+" => "+match[0])
        # check = input()
        # if check == "1":
        #     check.at[i, 'institution'] = match[0]
    i+=1
    print()
check.to_csv('College-Plans_updatedNames.csv')
'''

'''
# manual checking with state filter
for name in checkNames:
    if name not in fullNameList:
        nameList = []
        financesNames = against.loc[against['State'] == check.at[i, 'state']]
        #print(financesNames)
        for e in financesNames["Name"]:
            nameList.append(e)
        match = process.extract(name,nameList, scorer = fuzz.token_set_ratio)
        print("i: "+str(i)+" "+name+":")
        if len(match) != 0:
            for a in range(5):
                print(str(a+1)+": "+str(match[a]))
            number = input()
            if number == "1":
                check.at[i, 'institution'] = match[0][0]
            elif number == "2":
                check.at[i, 'institution'] = match[1][0]
            elif number == "3":
                check.at[i, 'institution'] = match[2][0]
            elif number == "4":
                check.at[i, 'institution'] = match[3][0]
            elif number == "5":
                check.at[i, 'institution'] = match[4][0]
    i+=1
check.to_csv('College-Plans_updatedNames2.csv')
'''