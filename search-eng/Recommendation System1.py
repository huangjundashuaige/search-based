"""
The International Genetically Engineered Machine Competition
Project : Recommendation System based on Word Vector and KDTree
Institution : SYSU
Team : SYSU_Software
Coders : FXY & DW 
All Right Reserved
"""


"""Import Libraries"""
# Manage the working directory
import os
# NLP-DL Library (Convert Key Words into Word Vectors)
import gensim
# Numpy is an efficient numerical library of python
import numpy as np
# We need the square root function
from math import sqrt
# We build KDTree of the Word Vectors for efficient search
#from sklearn.neighbors import KDTree
# We use Heap to implement the Top K index Search
import heapq
import operator
# Load Model
from pickle import load

# Http handle module
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib import parse



"""Change Current Directory"""
dire = "/mnt/igemmodel"
os.chdir(dire)
del dire


"""Load Data"""
Key_Words = np.load("Key Words_Final.npy")
Recommendation_Matrix = np.load("Reco_Final.npy")
Key_Team = np.load("Key_Team_Final.npy")
Teams = np.load("Team_Final.npy")
#Teams = Teams[0] #Bug Repair
Parts = np.load("Parts_Final.npy")

    
"""Load Model"""
Kdtree = load(open("kd_Final.sav",'rb'))
model = gensim.models.Word2Vec.load("wiki.en.text.model")


# Warning : The above should be done before openning the System to users
# Warning : Loading the Data and Model can be time consuming


"""Main Function (Open to the users)"""
def main(key_input, Interest) :
    """
    Interest is a list of integers indicating the users' taste
    """
    
    # The taste of users
    Taste = ['environment','disease','food','energy','hardware','software','industry','pollution','agriculture','art','heavy metal']
    
    #Some Given Constants
    M = 20 #Number of parts to recommend
    K = 5  #Number of Nearest Key Words
    T = 20  #Number of Teams to recommend
    bias = 0.001

    # Get Users' key word
    key_input = key_input.split()

    # Judge if the input is valid
    flag = 1
    for word in key_input:
        if word.lower() not in model:
            flag = 0
            break
    
    # Change the key word into Word Vector
    key_vector = []
    if flag == 1:
        key_vector = sum([ model.wv[word.lower()] for word in key_input])
    else:
        return "Invalid Input!"
    
    # Normalize the key_vector
    L2_Norm = sqrt(sum([item**2 for item in key_vector]))
    key_vector = key_vector/L2_Norm

    # Search the K Nearest Neighbors by KD tree
    dist,index = Kdtree.query([key_vector] , k = K )

    # Show the Similar Key Words to the user
    '''
    print(" ")
    print("Similar Key Words:")
    for i in range(len(index[0])) : 
        print( str(i+1) + "th Key Word: " + Key_Words[index[0][i]] )
    '''
    keyWords = [Key_Words[index[0][i]] for i in range(len(index[0]))]
    
    # Users' Portrait
    def L2(A,B):
        return sqrt( sum([ (A[i]-B[i])**2 for i in range(len(A))]) )
    
    Taste_Vector = []
    Portrait = []
    Portrait_Score = []
    if len(Interest) > 0:
        for intere in Interest:
            words = intere.split()
            WV_interest = sum( [ model.wv[word.lower()] for word in words ] )
            L2_Norm = sqrt(sum([item**2 for item in WV_interest]))
            WV_interest = WV_interest/L2_Norm
            Taste_Vector.append(WV_interest)
        
        for i in range(K):
            words = Key_Words[index[0][i]].split()
            WV_KEY = sum( [ model.wv[word.lower()] for word in words ] )
            L2_Norm = sqrt(sum([item**2 for item in WV_KEY]))
            WV_KEY = WV_KEY/L2_Norm
            
            scores = []
            for intere in range(len(Interest)):
                WV_interest = Taste_Vector[intere]
                distance = L2( WV_interest, WV_KEY )
                scores.append(distance)
                
            Portrait.append(scores)
        
        Portrait_Score = [ sum(row) for row in Portrait ]
    else:
        Portrait_Score = [ 1.0 for i in range(K) ]
    
    # Collaborative Filtering for Parts Recommendation (Based on Random Walk)
    Recommendation = [ Portrait_Score[i]*Recommendation_Matrix[ index[0][i] ]/(dist[0][i]+bias) for i in range(K) ]
    Parts_Score = [ sum(part) for part in zip(*Recommendation)]
    Parts_Recommendation = heapq.nlargest(M, enumerate(Parts_Score), key=operator.itemgetter(1))

    # Show the Recommended Parts
    '''
    print(" ")
    print("Recommanded Parts:")
    for i in range(M) : 
        print( str(i+1) + "th Part: " + Parts[ Parts_Recommendation[i][0] ] + "(Score: " + str(Parts_Recommendation[i][1]) + ")"  )
    '''
    parts = [Parts[Parts_Recommendation[i][0]] for i in range(M)]
    
    # Collaborative Filtering for Teams Recommendation (Based on Random Walk)
    Recommendation = [ Portrait_Score[i]*Key_Team[ index[0][i] ]/(dist[0][i]+bias) for i in range(K) ]
    Teams_Score = [ sum(team) for team in zip(*Recommendation)]
    Teams_Recommendation = heapq.nlargest(M, enumerate(Teams_Score), key=operator.itemgetter(1))

    # Show the Recommended Parts
    '''
    print(" ")
    print("Recommanded Teams:")
    for i in range(T) : 
        print( str(i+1) + "th Team: " + Teams[ Teams_Recommendation[i][0] ] + "(Score: " + str(Teams_Recommendation[i][1]) + ")"  )    
    '''
    teams = [Teams[Teams_Recommendation[i][0]] for i in range(T)]

    return {
        'keyWords': keyWords,
        'parts': parts,
        'teams': teams}

            
hostname = ''
hostport = 10086

class TinyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        path = self.path[2: ]
        params = parse.parse_qs(path)
        print("GET " + params['key'][0] + ' ' + str(params['interest'][0]))
        res = main(params['key'][0], json.loads(params['interest'][0]))
        self.wfile.write(json.dumps(res).encode())

if __name__=="__main__":
    server = HTTPServer((hostname, hostport), TinyServer)
    print("Start server")
    server.serve_forever()
