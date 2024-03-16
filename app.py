from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
app = Flask(__name__)

animal_disease_model = pickle.load(open('animal_disease.pkl','rb'))
diabetes_model = pickle.load(open('diabetes_model.pkl','rb'))
heart_model = pickle.load(open('heart_model.pkl','rb'))
#breast_model = pickle.load(open('breast_cancer.pkl','rb'))
symptoms_model = pickle.load(open('symptoms_based_detection.pkl','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/artical')
def artical():
	return render_template('articals.html')

@app.route('/human')
def human():
	return render_template('human.html')

@app.route('/animal')
def animal():
	return render_template('animal_disease_predict.html')

@app.route('/animal_prediction',methods=['GET','POST'])
def animal_prediction():
	sudden_death = request.form.get("sudden_death")
	blood_from_nose = request.form.get("blood_from_nose")
	trembling = request.form.get("trembling")
	difficult_breathing =request.form.get("difficult_breathing")
	blood_from_openings = request.form.get("blood_from_openings")
	fever=request.form.get("fever")
	loss_of_appetite =request.form.get("loss_of_appetite")
	dullness =request.form.get("dullness")
	swelling = request.form.get("swelling")
	recumnency = request.form.get("recumnency")
	profuse_salivation =request.form.get("profuse_salivation")
	vesicles = request.form.get("vesicles")
	lameness = request.form.get("lameness")
	change_in_behaviour = request.form.get("change_in_behaviour")
	furious =request.form.get("furious")
	dumbness =request.form.get("dumbness")
	nasal_discharge = request.form.get("nasal_discharge")
	eye_discharge =request.form.get("eye_discharge")
	haemorrage = request.form.get("haemorrage")
	lethargy = request.form.get("lethargy")
	enteritis = request.form.get("enteritis")
	abortion = request.form.get("abortion")
	no_breed = request.form.get("no_breed")
	unwillingness = request.form.get("unwillingness")
	stiffness = request.form.get("stiffness")
	eraction = request.form.get("eraction")
	mastication =request.form.get("mastication")
	paralysis =request.form.get("paralysis")
	encephalitis = request.form.get("encephalitis")
	septicaemia = request.form.get("septicaemia")
	infertility = request.form.get("infertility")
	nacrotic_foci = request.form.get("nacrotic_foci")
	diarrhea = request.form.get("diarrhea")
	weight_loss = request.form.get("weight_loss")
	shivering = request.form.get("shivering")
	drooling = request.form.get("drooling")
	excessive_urination = request.form.get("excessive_urination")

	animal_disease_data =(sudden_death,blood_from_nose,trembling,difficult_breathing,blood_from_openings,
												fever,loss_of_appetite,dullness,swelling,recumnency,profuse_salivation,vesicles,
												lameness,change_in_behaviour,furious,dumbness,nasal_discharge,eye_discharge,haemorrage,
												lethargy,enteritis,abortion,no_breed,unwillingness,stiffness,eraction,mastication,paralysis,
												encephalitis,septicaemia,infertility,nacrotic_foci,diarrhea,weight_loss,shivering,drooling,
												excessive_urination)

	animal_disease_data_num = np.array(animal_disease_data, dtype=float)  # convert using numpy
	# c = [float(i) for i in animal_disease_data]

	animal_disease_data_array = np.asarray(animal_disease_data_num)
	animal_disease_data_reshape = animal_disease_data_array.reshape(1,-1)
	animal_disease_prediction = animal_disease_model.predict(animal_disease_data_reshape)

	return render_template('animal_disease_predict.html',animal_disease_prediction = "The disease is {} ".format(animal_disease_prediction))


@app.route('/plant')
def plant():
	return render_template('plants.html')

@app.route('/diabetes')
def diabetes():
	return render_template('Diabetics.html')

@app.route('/diabetics_detect',methods=['GET','POST'])
def diabetics_detect():
	Pregnancies = request.form.get('Pregnancies')
	infertility = request.form.get('infertility')
	nacrotic_foci = request.form.get('nacrotic_foci')
	diarrhea = request.form.get('diarrhea')
	weight_loss = request.form.get('weight_loss')
	shivering = request.form.get('shivering')
	drooling = request.form.get('drooling')
	excessive_urination = request.form.get('excessive_urination')
	
	diabetes_data = (Pregnancies,infertility,nacrotic_foci,diarrhea,weight_loss,shivering,drooling,excessive_urination)
	dibetes_array = np.asarray(diabetes_data)
	diabetes_reshape = dibetes_array.reshape(1,-1)
	diabetes_prediction = diabetes_model.predict(diabetes_reshape)
	print(diabetes_prediction)
	if diabetes_prediction == 1:
		return render_template('Diabetics.html',Diabetics_disease_prediction="person has a dibetics")
	else:
		return render_template('Diabetics.html',Diabetics_disease_prediction="person has Non-dibetics")


@app.route('/heart')
def heart():
	return render_template('Heart.html')

@app.route('/heart_detection',methods=['POST'])
def heart_detection():
	septicaemia = request.form.get('septicaemia')
	infertility = request.form.get('infertility')
	nacrotic_foci = request.form.get('nacrotic_foci')
	diarrhea = request.form.get('diarrhea')
	weight_loss = request.form.get('weight_loss')
	shivering = request.form.get('shivering')
	drooling = request.form.get('drooling')
	excessive_urination = request.form.get('excessive_urination')
	septicaemiaa = request.form.get('septicaemiaa')
	infertilityy = request.form.get('infertilityy')
	nacrotic_fociii = request.form.get('nacrotic_fociii')
	diarrheaaa = request.form.get('diarrheaaa')
	weight_lossss=request.form.get('weight_lossss')

	heart_data = (septicaemia,infertility,nacrotic_foci,diarrhea,weight_loss,shivering,drooling,excessive_urination,septicaemiaa,infertilityy,nacrotic_fociii,diarrheaaa,weight_lossss)
	heart_array = np.asarray(heart_data,dtype=float)
	heart_reshape = heart_array.reshape(1,-1)
	heart_prediction = heart_model.predict(heart_reshape)
	print(heart_prediction)
	if heart_prediction == 1:
		return render_template('Heart.html',Heart_disease_prediction="person has a heart disease")
	else:
		return render_template('Heart.html',Heart_disease_prediction="person has No heart disease")


	
	#return render_template('Heart.html')

@app.route('/breast')
def breast():
	return render_template('breastCancer.html')

@app.route('/breast_detection',methods=['GET','POST'])
def breast_detection():
	swelling = request.form.get('swelling')
	recumnency = request.form.get('recumnency')
	profuse_salivation = request.form.get('profuse_salivation')
	vesicles = request.form.get('vesicles')
	lameness = request.form.get('lameness')
	change_in_behaviour = request.form.get('change_in_behaviour')
	furious = request.form.get('furious')
	dumbness = request.form.get('dumbness')
	nasal_discharge = request.form.get('nasal_discharge')
	eye_discharge = request.form.get('eye_discharge')
	haemorrage = request.form.get('haemorrage')
	lethargy = request.form.get('lethargy')
	enteritis = request.form.get('enteritis')
	abortion = request.form.get('abortion')
	no_breed = request.form.get('no_breed')
	unwillingness = request.form.get('unwillingness')
	stiffness = request.form.get('stiffness')
	eraction = request.form.get('eraction')
	mastication = request.form.get('mastication')
	paralysis = request.form.get('paralysis')
	encephalitis = request.form.get('encephalitis')
	septicaemia = request.form.get('septicaemia')
	infertility = request.form.get('infertility')
	nacrotic_foci = request.form.get('nacrotic_foci')
	diarrhea = request.form.get('diarrhea')
	weight_loss = request.form.get('weight_loss')
	shivering = request.form.get('shivering')
	drooling = request.form.get('drooling')
	excessive_urination = request.form.get('excessive_urination')
	dullness = request.form.get('dullness')

	breast_data = (swelling,recumnency,profuse_salivation,vesicles,lameness,change_in_behaviour,furious,dumbness,nasal_discharge,eye_discharge,haemorrage,lethargy,enteritis,abortion,no_breed,unwillingness,stiffness,eraction,mastication,paralysis,encephalitis,septicaemia,infertility,nacrotic_foci,diarrhea,weight_loss,shivering,drooling,excessive_urination,dullness)
	breast_array = np.asarray(breast_data,dtype=float)
	breast_reshape = breast_array.reshape(1,-1)
	#breast_fit =scaler.fit(breast_reshape)
	#breast_std = scaler.transform(breast_reshape)
	breast_prediction = "Error occured please back after some time"
	print(breast_prediction)

	return "prediction = {}".format(breast_prediction)
	# if heart_prediction == 1:
	# 	return render_template('Heart.html',Heart_disease_prediction="person has a heart disease")
	# else:
	# 	return render_template('Heart.html',Heart_disease_prediction="person has No heart disease")



@app.route('/symp')
def symp():
	return render_template('Symptom.html')

@app.route('/symp_detection',methods=['POST'])
def symp_detection():
	s1 = request.form.get('s1')
	s2 = request.form.get('s2')
	s3 = request.form.get('s3')
	s4 = request.form.get('s4')
	s5 = request.form.get('s5')
	s6 = request.form.get('s6')
	s7 = request.form.get('s7')
	s8 = request.form.get('s8')
	s9 = request.form.get('s9')
	s10 = request.form.get('s10')
	s11 = request.form.get('s11')
	s12 = request.form.get('s12')
	s13 = request.form.get('s13')
	s14 = request.form.get('s14')
	s15 = request.form.get('s15')
	s16 = request.form.get('s16')
	s17 = request.form.get('s17')
	s18 = request.form.get('s18')
	s19 = request.form.get('s19')
	s20 = request.form.get('s20')
	s21 = request.form.get('s21')
	s22 = request.form.get('s22')
	s23 = request.form.get('s23')
	s24 = request.form.get('s24')
	s25 = request.form.get('s25')
	s26 = request.form.get('s26')
	s27 = request.form.get('s27')
	s28 = request.form.get('s28')
	s29 = request.form.get('s29')
	s30 = request.form.get('s30')
	s31 = request.form.get('s31')
	s32 = request.form.get('s32')
	s33 = request.form.get('s33')
	s34 = request.form.get('s34')
	s35 = request.form.get('s35')
	s36 = request.form.get('s36')
	s37 = request.form.get('s37')
	s38 = request.form.get('s38')
	s39 = request.form.get('s39')
	s40 = request.form.get('s40')
	s41 = request.form.get('s41')
	s42 = request.form.get('s42')
	s43 = request.form.get('s43')
	s44 = request.form.get('s44')
	s45 = request.form.get('s45')
	s46 = request.form.get('s46')
	s47 = request.form.get('s47')
	s48 = request.form.get('s48')
	s49 = request.form.get('s49')
	s50 = request.form.get('s50')
	s51 = request.form.get('s51')
	s52 = request.form.get('s52')
	s53 = request.form.get('s53')
	s54 = request.form.get('s54')
	s55 = request.form.get('s55')
	s56 = request.form.get('s56')
	s57 = request.form.get('s57')
	s58 = request.form.get('s58')
	s59 = request.form.get('s59')
	s60 = request.form.get('s60')
	s61 = request.form.get('s61')
	s62 = request.form.get('s62')
	s63 = request.form.get('s63')
	s64 = request.form.get('s64')
	s65 = request.form.get('s65')
	s66 = request.form.get('s66')
	s67 = request.form.get('s67')
	s68 = request.form.get('s68')
	s69 = request.form.get('s69')
	s70 = request.form.get('s70')
	s71 = request.form.get('s71')
	s72 = request.form.get('s72')
	s73 = request.form.get('s73')
	s74 = request.form.get('s74')
	s75 = request.form.get('s75')
	s76 = request.form.get('s76')
	s77 = request.form.get('s77')
	s78 = request.form.get('s78')
	s79 = request.form.get('s79')
	s80 = request.form.get('s80')
	s81 = request.form.get('s81')
	s82 = request.form.get('s82')
	s83 = request.form.get('s83')
	s84 = request.form.get('s84')
	s85 = request.form.get('s85')
	s86 = request.form.get('s86')
	s87 = request.form.get('s87')
	s88 = request.form.get('s88')
	s89 = request.form.get('s89')
	s90 = request.form.get('s90')
	s91 = request.form.get('s91')
	s92 = request.form.get('s92')
	s93 = request.form.get('s93')
	s94 = request.form.get('s94')
	s95 = request.form.get('s95')
	s96 = request.form.get('s96')
	s97 = request.form.get('s97')
	s98 = request.form.get('s98')
	s99 = request.form.get('s99')
	s100 = request.form.get('s100')
	s101 = request.form.get('s101')
	s102 = request.form.get('s102')
	s103 = request.form.get('s103')
	s104 = request.form.get('s104')
	s105 = request.form.get('s105')
	s106 = request.form.get('s106')
	s107 = request.form.get('s107')
	s108 = request.form.get('s108')
	s109 = request.form.get('s109')
	s110 = request.form.get('s110')
	s111 = request.form.get('s111')
	s112 = request.form.get('s112')
	s113 = request.form.get('s113')
	s114 = request.form.get('s114')
	s115 = request.form.get('s115')
	s116 = request.form.get('s116')
	s117 = request.form.get('s117')
	s118 = request.form.get('s118')
	s119 = request.form.get('s119')
	s120 = request.form.get('s120')
	s121 = request.form.get('s121')
	s122 = request.form.get('s122')
	s123 = request.form.get('s123')
	s124 = request.form.get('s124')
	s125 = request.form.get('s125')
	s126 = request.form.get('s126')
	s127 = request.form.get('s127')
	s128 = request.form.get('s128')
	s129 = request.form.get('s129')
	s130 = request.form.get('s130')
	s131 = request.form.get('s131')
	s132 = request.form.get('s132')

	symptom_data = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40,s41,s42,s43,s44,s45,s46,s47,s48,s49,s50,s51,s52,s53,s54,s55,s56,s57,s58,s59,s60,s61,s62,s63,s64,s65,s66,s67,s68,s69,s70,s71,s72,s73,s74,s75,s76,s77,s78,s79,s80,s81,s82,s83,s84,s85,s86,s87,s88,s89,s90,s91,s92,s93,s94,s95,s96,s97,s98,s99,s100,s101,s102,s103,s104,s105,s106,s107,s108,s109,s110,s111,s112,s113,s114,s115,s116,s117,s118,s119,s120,s121,s122,s123,s124,s125,s126,s127,s128,s129,s130,s131,s132]
	#symptom_data.fillna(symptom_data.mean())
	sym_data_num = np.array(symptom_data, dtype=float) 
	sym_array = np.asarray(sym_data_num)
	sym_reshape = sym_array.reshape(1,-1)
	sym_detection = symptoms_model.predict(sym_reshape)
	print(sym_detection)
	return render_template('Symptom.html',symptom_disease_prediction = "The disease is {} ".format(sym_detection))

if __name__ == '__main__':
	app.run()




	