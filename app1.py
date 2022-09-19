import os
from PIL import Image
import pytesseract
import cv2
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template,Blueprint,current_app
import os
global filename1
import io

dir_path = os.path.dirname(os.path.realpath(__file__))

#UPLOAD_FOLDER = 'C:/Users/Administrator/Desktop/Flask/uploads/'
#app_file1= Flask(__name__, template_folder='templates')
app_file1 = Blueprint('app_file1',__name__)




def getFile(filename1):
    
    #!/usr/bin/env python
    # coding: utf-8
    
    # In[2]:
    
    
    import cv2
    import numpy as np
    image = cv2.imread(filename1)
    
    
    # In[3]:
    
    
    
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    cv2.waitKey(0)
    
    
    # In[4]:
    
    
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('second',thresh)
    cv2.waitKey(0)
    
    
    # In[5]:
    
    
    #dilation
    kernel = np.ones((5,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('dilated',img_dilation)
    cv2.waitKey(0)
    
    
    # In[6]:
    
    
    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # In[7]:
    
    
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    
    # In[8]:
    
    
    sorted_ctrs.reverse()
    
    
    # In[9]:
    
    
    print(len(sorted_ctrs))
    
    
    # In[10]:
    
    
    
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
    
        # Getting ROI
        roi = image[y:y+h, x:x+w]
    
        # show ROI
        cv2.imwrite('C:/Users/GAUTAMI/Desktop/letters/Lines/line{}.jpg'.format(i),roi)
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        cv2.waitKey(0)
        
    
    
    # In[11]:
    
    
    cv2.imshow('marked areas',image)
    cv2.waitKey(0)
    
    
    # In[12]:
    
    
    import os
    count_lines = 0
    
    
    d = "C:/Users/GAUTAMI/Desktop/letters/Lines"
    for path in os.listdir(d):
        if os.path.isfile(os.path.join(d, path)):
            count_lines += 1
            
    print(count_lines)
    
    
    # In[13]:
    
    
    import os
    count=0
    #word segmentation
    for i in range(0,count_lines):
        image = cv2.imread('C:/Users/GAUTAMI/Desktop/letters/Lines/line{}.jpg'.format(i))
        
        #Make seperate word folders for seperate lines
        os.mkdir('C:/Users/GAUTAMI/Desktop/letters/Words/line{}'.format(i))
        
        
        #cv2.imshow('image',image)
        #print(image.size)    
        
        #grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray',gray)
        cv2.waitKey(0)
        
        #binary
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        #cv2.imshow('second',thresh)
        cv2.waitKey(0)
        
        #dilation
        kernel = np.ones((5,5), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        #cv2.imshow('dilated',img_dilation)
        cv2.waitKey(0)
        
        #find contours
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_ctrs_words = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        
        
    
        for k, ctr in enumerate(sorted_ctrs_words):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
    
                
    
            # Getting ROI
            roi = image[y:y+h, x:x+w]
                
    
                
            # show ROI
            cv2.imwrite('C:/Users/GAUTAMI/Desktop/letters/Words/line{}/word{}.jpg'.format(i,k),roi)
            cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
            cv2.waitKey(0)
    
            #cv2.imshow('marked areas',image)
            cv2.waitKey(0)
            count=count+1
    
    
    # In[14]:
    
    
    import os
    count_word_folders = len(next(os.walk('C:/Users/GAUTAMI/Desktop/letters/Words'))[1])
    print(count_word_folders)
    
    
    # In[15]:
    
    
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import pylab
    import sys
    
    
    #a = sys.argv[1:][0]
    #flag = int(sys.argv[1:][1])
    
    
    class letter_finder:
            
        def __init__(self,img):
            self.img = cv2.imread(img)
            self.img = cv2.resize(self.img,(0,0),fx=10,fy=5)
            avg = np.average(self.img)
            self.b_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
            self.rows = self.img.shape[0]
            self.cols = self.img.shape[1]
            
        def threshold(self,thr_val):
            (t,thr_img) = cv2.threshold(self.b_img,thr_val,255,cv2.THRESH_BINARY)
            self.thr_img = thr_img 
    
    
        def find_line(self):
            count_matrix = []
            for y in range(0,self.thr_img.shape[0]-1):
                count = 0
                for x in range(0,self.thr_img.shape[1]-1):
                    if self.thr_img[y][x] == 0:
                        count += 1
                count_matrix.append(count)
            for i in range(len(count_matrix)-2,0,-1):
                if count_matrix[i] > int(self.cols/5):
                    #print(count_matrix[i])
                    bottom_line = i 
                    break
            #print(count_matrix)
            #print('b',count_matrix[bottom_line])
            return (count_matrix,bottom_line)
    
        def remove_line(self,count_matrix):
            margin = 11
            y_line = count_matrix[0].index(max(count_matrix[0]))
            #print(count_matrix[1])
            upper_img = self.thr_img[0:(y_line - margin),0:self.cols]
            lower_img = self.thr_img[(y_line + margin):count_matrix[1],0:self.cols]
            
            final_image = np.concatenate((upper_img,lower_img),axis=0)
            #print('image has been formed without line')
            self.final_image = final_image
    
        def show_letters(self):
            for i in range(0,len(self.letter_matrix)-2):
                cv2.rectangle(self.img,(self.letter_matrix[i],0),(self.letter_matrix[i+1],self.cols),0,2)
            #print('letters are drawn')
    
        def count_region(self,count_matrix,pos,r):
            count = sum(count_matrix[pos-r:pos+r])
            return count
    
        def find_letters(self,x1,y1,x2,y2):
            count_matrix = []
            letter_matrix = []
            letter_matrix.append(x1 + 10)
            for x in range(x1,x2):
                count = 0
                for y in range(y1,y2):
                    if self.final_image[y][x] == 0:
                        count += 1
                count_matrix.append(count)
            #print(count_matrix)
            x = x1 + 80
            while x < len(count_matrix)-2:
                if self.count_region(count_matrix,x,3) < 2:
                    if (x + x1) not in letter_matrix:
                        letter_matrix.append(x + x1 + 10)
                        x += 40
                x += 1
            for first,second in zip(letter_matrix,letter_matrix[1:]):
                #print(second-first)
                if second - first < 50:
                #print('removing')
                    letter_matrix.remove(first)
            #print('letters have been found')
            #print('letter_matrix',letter_matrix)
            self.letter_matrix = letter_matrix
            #print('letter_matrix',self.letter_matrix)
            self.no_words = len(letter_matrix)-1
            self.count_matrix = count_matrix
    
        def show_letters(self):#draws boxes around letters
            for i in range(0,len(self.letter_matrix)-1):
                cv2.rectangle(self.img,(self.letter_matrix[i],0),(self.letter_matrix[i+1],self.rows),0,1)
            #print('letters are drawn')	
    
        def resize_image(self,x,y):
            self.img = cv2.resize(self.img,(0,0),fx=x,fy=y)
    
        def show_image(self):
            cv2.imshow('letters',self.img)
            cv2.waitKey(0)
    
        def show_cropped_image(self):
            cv2.imshow('letters',self.final_image)
            cv2.waitKey(0)
    
        def plot_intensity(self):
            x = []
            for i in range(len(self.count_matrix)):
                x.append(i)
    
            plt.plot(x,self.count_matrix)
            plt.show()
    
        def crop_letters(self,word_index):
            for x in range(0,len(self.letter_matrix)-1):
                letter = self.img[0:self.rows , self.letter_matrix[x]:self.letter_matrix[x+1]]
                letter = cv2.resize(letter,(0,0),fx=0.2,fy=0.2)
                cv2.imwrite('C:/Users/GAUTAMI/Desktop/letters/Character/line{}/word{}/char'.format(i,j)+str(word_index)+str(x)+'.jpg',letter)
                
        
            #print('letters have been stored...')
    
        def store_cropped_letters(self,word_index):
            y = self.find_line()
            self.remove_line(y)
            self.find_letters(0,0,self.cols,self.final_image.shape[0])
            self.crop_letters(word_index)
    
    
    for i in range(0, count_word_folders):
        
        import os
        count_words = 0
        d = "C:/Users/GAUTAMI/Desktop/letters/Words/line{}".format(i)
        for path in os.listdir(d):
            if os.path.isfile(os.path.join(d, path)):
                count_words += 1
        
        print(count_words)
        os.mkdir('C:/Users/GAUTAMI/Desktop/letters/Character/line{}'.format(i))
                
        for j in range(0,count_words):
    
            #im = 'C:/Users/Administrator/Desktop/DEMO_LINE' + a + '.jpg'
            im='C:/Users/GAUTAMI/Desktop/letters/Words/line{}/word{}.jpg'.format(i,j)
            #import os
            #print(i,'',j)
            os.mkdir('C:/Users/GAUTAMI/Desktop/letters/Character/line{}/word{}'.format(i,j))
            #cv2.imshow('image',im)
            image = letter_finder(im)
            img = cv2.imread(im)
            #img = cv2.resize(img,(0,0),fx=10,fy=5)
            #avg1 = np.average(img)
            #med1 = np.median(img)
            b_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            avg = np.average(b_img)
            #ratio = 190.0/avg
            #med = np.median(b_img)
            #print('avg',avg,avg1)
            #print('med',med,med1)
            #b_img += 50
            #cv2.imshow('1',b_img)
            #cv2.waitKey(0)
    
            #########################################
            image.threshold(avg - 40)
            y = image.find_line()
            #image.remove_line(y)
            #image.show_cropped_image()
            #image.find_letters(0,0,image.cols,image.final_image.shape[0])
            #image.plot_intensity()
            #image.show_letters()
            #image.show_image()
            image.store_cropped_letters(0)
    
    
    
    
    
    
    #Prediction
    
    import tensorflow as tf
    classifier = tf.keras.models.load_model('C:/Users/GAUTAMI/Downloads/cnn_ocr_32_(tf=2.1.0).h5')
    
    classifier.summary()
    training_indices={'0': 0,
    '1': 1,'10': 2,'100': 3,'101': 4,
    '102': 5,'103': 6,'104': 7,'105': 8,
    '106': 9,'107': 10,'108': 11,'109': 12,
    '11': 13,'110': 14,'111': 15,'113': 16,
    '114': 17,'115': 18,'116': 19,'117': 20,
    '118': 21,'119': 22,'12': 23,'120': 24,
    '121': 25,'122': 26,'123': 27,'124': 28,
    '125': 29,'126': 30,'127': 31,'128': 32,
    '129': 33,'13': 34,'130': 35,'131': 36,
    '132': 37,'133': 38,'134': 39,'135': 40,
    '137': 41,'138': 42,'139': 43,'14': 44,
    '140': 45,'142': 46,'143': 47,'144': 48,
    '145': 49,'146': 50,'147': 51,'148': 52,
    '149': 53,'15': 54,'150': 55,'151': 56,
    '152': 57,'153': 58,'154': 59,'155': 60,
    '156': 61,'157': 62,'158': 63,'159': 64,
    '16': 65,'160': 66,'161': 67,'162': 68,
    '163': 69,'164': 70,'165': 71,'166': 72,
    '167': 73,'168': 74,'169': 75,'17': 76,
    '170': 77,'171': 78,'172': 79,'173': 80,
    '174': 81,'175': 82,'176': 83,'177': 84,
    '178': 85,'179': 86,'18': 87,'180': 88,
    '181': 89,'182': 90,'183': 91,'184': 92,
    '185': 93,'186': 94,'188': 95,'189': 96,'19': 97,
    '190': 98,'191': 99,'192': 100,'193': 101,
    '194': 102,'195': 103,'196': 104,'197': 105,
    '198': 106,'199': 107,'2': 108,'20': 109,
    '200': 110,'201': 111,'202': 112,'203': 113,
    '204': 114,'205': 115,'206': 116,'207': 117,
    '208': 118,'209': 119,'21': 120,'210': 121,
    '211': 122,'212': 123,'213': 124,'214': 125,
    '215': 126,'216': 127,'217': 128,'218': 129,
    '219': 130,'22': 131,'220': 132, '221': 133,'222': 134,
    '223': 135,'224': 136,'225': 137,'226': 138,
    '227': 139,'228': 140,'229': 141,'23': 142,
    '230': 143,'231': 144,'232': 145,'233': 146,
    '234': 147,'235': 148,'236': 149,'238': 150,
    '239': 151,'24': 152,'240': 153,'241': 154,
    '242': 155,'243': 156,'246': 157,'247': 158,
    '248': 159,'249': 160,'25': 161,'250': 162,
    '251': 163,'252': 164,'253': 165,'254': 166,
    '255': 167,'256': 168,'258': 169,'259': 170,
    '26': 171,'260': 172,'261': 173,'262': 174,
    '263': 175,'264': 176,'265': 177,'266': 178,
    '267': 179,'268': 180,'269': 181,'27': 182,
    '270': 183,'271': 184,'273': 185,'274': 186,
    '275': 187,'276': 188,'277': 189,'278': 190,
    '279': 191,'28': 192,'280': 193,'281': 194,
    '282': 195,'283': 196,'284': 197,'285': 198,
    '286': 199,'287': 200,'288': 201,'289': 202,
    '29': 203,'291': 204,'292': 205,'293': 206,
    '294': 207,'296': 208,'297': 209,'298': 210,
    '299': 211,'3': 212,'30': 213,'300': 214,
    '301': 215,'302': 216,'303': 217,'304': 218,
    '305': 219,'306': 220,'307': 221,'308': 222,
    '31': 223,'310': 224,'311': 225,'312': 226,
    '313': 227,'314': 228,'315': 229,'316': 230,
    '317': 231,'318': 232,'319': 233,'32': 234,
    '320': 235,'321': 236,'322': 237,'323': 238,
    '324': 239,'325': 240,'326': 241,'327': 242,
    '328': 243,'329': 244,'33': 245,'330': 246,
    '331': 247,'332': 248,'333': 249,'334': 250,
    '335': 251,'336': 252,'337': 253,'338': 254,
    '339': 255,'34': 256,'340': 257,'341': 258,
    '342': 259,'343': 260,'344': 261,'345': 262,
    '346': 263,'347': 264,'348': 265,'349': 266,
    '35': 267,'350': 268,'352': 269,'353': 270,
    '356': 271,'357': 272,'358': 273,'359': 274,
    '36': 275,'360': 276,'361': 277,'362': 278,
    '363': 279,'364': 280,'365': 281,'366': 282,
    '367': 283,'368': 284,'369': 285,'370': 286,
    '371': 287,'372': 288,'373': 289,'374': 290,
    '375': 291,'376': 292,'377': 293,'378': 294,
    '379': 295,'380': 296,'382': 297,'383': 298,
    '384': 299,'385': 300,'386': 301,'387': 302,
    '388': 303,'389': 304,'39': 305,'390': 306,
    '391': 307,'392': 308,'393': 309,'394': 310,
    '395': 311,'396': 312,'397': 313,'398': 314,
    '399': 315,'4': 316,'40': 317,'400': 318,
    '401': 319,'402': 320,'403': 321,'404': 322,
    '405': 323,'406': 324,'407': 325,'408': 326,
    '409': 327,'41': 328,'410': 329,'411': 330,
    '412': 331,'413': 332,'414': 333,'415': 334,
    '416': 335,'417': 336,'418': 337,'419': 338,
    '42': 339,'420': 340,'421': 341,'43': 342,
    '432': 343,'44': 344,'444': 345,'445': 346,
    '446': 347,'447': 348,'448': 349,'449': 350,
    '45': 351,'450': 352,'451': 353,'452': 354,
    '453': 355,'454': 356,'455': 357,'456': 358,
    '457': 359,'458': 360,'459': 361,'46': 362,
    '460': 363,'461': 364,'462': 365,'463': 366,
    '464': 367,'465': 368,'466': 369,'467': 370,
    '468': 371,'469': 372,'47': 373,'470': 374,
    '471': 375,'472': 376,'473': 377,'474': 378,
    '475': 379,'476': 380,'477': 381,'478': 382,
    '479': 383,'48': 384,'480': 385,'481': 386,
    '482': 387,'483': 388,'484': 389,'485': 390,
    '486': 391,'487': 392,'488': 393,'489': 394,
    '49': 395,'490': 396,'491': 397,'492': 398,
    '493': 399,'494': 400,'495': 401,'496': 402,
    '497': 403,'498': 404,'499': 405,'5': 406,
    '50': 407,'500': 408,'501': 409,'502': 410,
    '503': 411,'504': 412,'505': 413,'506': 414,
    '507': 415,'508': 416,'509': 417,'51': 418,
    '510': 419,'511': 420,'512': 421,'513': 422,
    '514': 423,'515': 424,'516': 425,'517': 426,
    '518': 427,'52': 428,'53': 429,'54': 430,
    '55': 431,'56': 432,'57': 433,'58': 434,'59': 435,
    '6': 436,'60': 437,'61': 438,'62': 439,'63': 440,'64': 441,'65': 442,'66': 443,'67': 444,'68': 445,'69': 446,'7': 447,
    '70': 448,'71': 449,'72': 450,'73': 451,'74': 452,'75': 453,'76': 454,'77': 455,'78': 456,'8': 457,'80': 458,'81': 459,'82': 460,'83': 461,'84': 462,'85': 463,'86': 464,'87': 465,'88': 466,'89': 467,'9': 468,'90': 469,'91': 470,'92': 471,'94': 472,'95': 473,'96': 474,'97': 475,'98': 476,'99': 477}
    
    import pandas as pd
    
    # Importing the dataset
    dataset = pd.read_csv('C:/Users/GAUTAMI/Desktop/Final_OCR/OCR_(3 layers)/Final_Mapping.csv')
    X = dataset.iloc[:,0].values
    Y = dataset.iloc[:,1].values
    
    
    import os
    line_folder_count=len(next(os.walk('C:/Users/GAUTAMI/Desktop/letters/Character'))[1])
    
    import io
    with io.open("tes_op.txt", "w", encoding="utf-8") as f:
        for i in range(1, line_folder_count):
            word_folder_count=len(next(os.walk('C:/Users/GAUTAMI/Desktop/letters/Character/line{}'.format(i)))[1])
            
            for j in range(0, word_folder_count):
                char_count= len(next(os.walk('C:/Users/GAUTAMI/Desktop/letters/Character/line{}/word{}'.format(i,j)))[2]) #dir is your directory path as string
                
                for k in range(0, char_count):
                    im='C:/Users/GAUTAMI/Desktop/letters/Character/line{}/word{}/char0{}.jpg'.format(i,j,k)
                    
                    import numpy as np
                    import cv2
                    from keras.preprocessing import image
                
                    test_image = image.load_img(im ,target_size = (32, 32))
                    
                    
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis = 0)
                    result = classifier.predict(test_image)
                    y1=np.argmax(result, axis=-1)
            
                    letter= int(list(training_indices.keys())[list(training_indices.values()).index(y1)])
                    
                    
                    #print(Y[letter])   
                    
                    f.write(Y[letter])
                    
                f.write(" ")
                
            f.write("\n")
        
    return('tes_op.txt')

@app_file1.route("/")
def home():
    return render_template('home.html')

@app_file1.route('/handleUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            filename1 =(os.path.join('uploads/', filename))
            filename2= file.save(filename1)
            res=getFile(filename1)
            print("saved file successfully")
      #send file name as parameter to downlad
            return redirect('/downloadfile/'+res) 
    return render_template('home.html')

# Download API
@app_file1.route("/downloadfile/<res>", methods = ['GET'])
def download_file(res):
    return render_template('downloads.html',value=res)

@app_file1.route('/return-files/<res>')
def return_files_tut(res):
    return send_file('uploads/'+res, as_attachment=True, attachment_filename='')
    return render_template('home.html')


