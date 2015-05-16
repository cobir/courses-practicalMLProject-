library(caret)

set.seed(125);
pmltrain<- read.csv("pml-training.csv");

# predict the classe variable (factor)

# training split 
inTrain = createDataPartition(pmltrain$classe, p = 3/4)[[1]];
training = pmltrain[ inTrain,];
testing = pmltrain[-inTrain,];

sapply(training,class);

#
#
# there are many features, most of them numeric; are some of them correlated?
# 


# this function take the training/testing input data frame
# and removes X, factor and NA columns
remove_facor_na<-function (d)
{
  features<-d[,-3]
  
  # make new_window numeric
  features$new_window<-as.numeric(features$new_window);
  
  numdf<-features[sapply(features,function(c) !is.factor(c))];
  
  numdfnn<-numdf[,colSums(is.na(numdf)) == 0];
  
  numdfnn[,-1];
}

tr<-remove_facor_na(training);

M<-abs(cor(tr));
diag(M)<-0;
which(M>0.9,arr.ind=T);

pp<-preProcess(tr,method="pca");
trPC<-predict(pp,tr);


library(doParallel)

c1<-makeCluster(3);
registerDoParallel(c1);
system.time(mf1<-train(training$classe ~., method="rf",data=trPC));


stopCluster(c1);


tst<-remove_facor_na(testing);
tstPC<-predict(pp,tst);


pr1<-predict(mf1, tstPC);
confusionMatrix(testing$classe,pr1);


# ---------------------------------------------
# generate text files for submission

pmltest<- read.csv("pml-testing.csv");

modeltst<-remove_facor_na(pmltest);
modeltestPC<-predict(pp,modeltst[,-56]);
predictions<-predict(mf1, modeltestPC);


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}



pml_write_files(predictions);


#------------------------------------------------
