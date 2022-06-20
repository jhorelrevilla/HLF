import numpy as np
from zoo.orca import init_orca_context, stop_orca_context
from pyspark.sql import SparkSession
from zoo.pipeline.api.keras.optimizers import Adam
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers.core import Dense, Activation
from bigdl.optim.optimizer import EveryEpoch, Loss, TrainSummary, ValidationSummary
from zoo.pipeline.nnframes import *
from zoo.common.nncontext import *
from zoo.pipeline.api.keras.objectives import CategoricalCrossEntropy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

spark=SparkSession.builder\
.appName("HLF")\
.master("yarn")\
.getOrCreate()

#Leer parquet
PATH = "hdfs://hadoop-master:9000/sparkCern/"
trainDF = spark.read.format('parquet')\
        .load(PATH + 'trainUndersampled_HLF_features')\
        .select(['HLF_input', 'encoded_label'])
        
testDF = spark.read.format('parquet')\
        .load(PATH + 'testUndersampled_HLF_features')\
        .select(['HLF_input', 'encoded_label'])

#Usar el 0.01
fraction=0.001
trainDF = trainDF.sample(fraction=fraction, seed=42)#3422
testDF = testDF.sample(fraction=fraction, seed=42)#858

init_nncontext("clasificador")

#Modelo red neuronal
model = Sequential()
model.add(Dense(50, input_shape=(14,), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

estimator = NNEstimator(model, CategoricalCrossEntropy()) \
.setOptimMethod(Adam()) \
.setBatchSize(32) \
.setMaxEpoch(10) \
.setFeaturesCol("HLF_input") \
.setLabelCol("encoded_label") \
.setValidation(trigger=EveryEpoch() , val_df=testDF,val_method=[Loss(CategoricalCrossEntropy())], batch_size=32)

appName = "Clasificador HLF"
logDir = "/tmp"
trainSummary = TrainSummary(log_dir=logDir,app_name=appName)
estimator.setTrainSummary(trainSummary)
valSummary = ValidationSummary(log_dir=logDir,app_name=appName)
estimator.setValidationSummary(valSummary)

trained_model = estimator.fit(trainDF)

#Plotear 1 imagen
plt.style.use('seaborn-darkgrid')
loss = np.array(trainSummary.read_scalar("Loss"))
val_loss = np.array(valSummary.read_scalar("Loss"))
plt.plot(loss[:,0], loss[:,1], label="Training loss")
plt.plot(val_loss[:,0], val_loss[:,1], label="Validation loss", color='crimson', alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title("HLF classifier loss")
plt.savefig('resultado1.png')

#guardar modelo
trained_model.save("clasificador")
#Entrenamiento
predDF = trained_model.transform(testDF)

#Obtener Roc
y_pred = np.asarray(predDF.select("prediction").collect())
y_true = np.asarray(testDF.select('encoded_label').rdd.map(lambda row: np.asarray(row.encoded_label)).collect())
y_pred = np.squeeze(y_pred)
y_pred.shape

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#plotear Roc y AUC
plt.figure()
plt.plot(fpr[0], tpr[0], lw=2, label='HLF classifier (AUC) = %0.4f' % roc_auc[0])
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background Contamination (FPR)')
plt.ylabel('Signal Efficiency (TPR)')
plt.title('$tt$ selector')
plt.legend(loc="lower right")
plt.savefig('resultado2.png')


