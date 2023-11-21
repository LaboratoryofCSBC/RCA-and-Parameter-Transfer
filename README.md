# RCA-and-Parameter-Transfer

My thesis "Parameter Transfer and Riemannian Space Coordinate Alignment for EEG Intention Recognition" related code.

If you think this repository is useful, please cite it in your papers and other publications!



RCA:

	covariance_matrix: Calculate the covariance matrix
 
	mianRCA: Perform Riemannian space coordinate alignment

ParameterTransfer:

	Step 1: First execute getWb.py to get the dataset parameter W, parameter b and the corresponding x. 
 		(To minimize computation, getWb.py is executed only once. All subjects are computed here, 
  		and subsequently in getWbpred.py the parameter W and b of the target subject will be eliminated.)
 
 	Step 2: Selection of target subjects first. Execute train_test_split.py to divide the dataset to 
  		ensure that the same training and test sets are used in subsequent operations.
	
 	Step 3: Specify the target subject. Execute getWbpred.py to predict the classifier parameters for 
  		the target subject based on the parameters W,b and corresponding x in the source domain. 
    		(Here the parameters W,b obtained from the target domain in the first step are eliminated 
      		because the target domain is unknown.)
	
 	Step 4: Specify the target subject.Execute test.py to get the SPT results and Softmax results. 
  		(In order to ensure the consistency of parameters, please execute test.py, do not execute 
    		SPT.py separately, parameter settings are unified in SPT.py.)
