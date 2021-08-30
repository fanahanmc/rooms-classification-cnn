# STAT40970 Machine Learning & AI Assignment 3

## 1. Encode the images in RGB tensors of width/height of 128 x 128.

setwd("/Users/fanahanmcsweeney/Projects/02-UCD-Spring-Trimester/STAT40970-MLAI/Assignment3/")

library("jpeg") # to load pictures
library("knitr")
library("keras")

# print classification table and overall classification accuracy
class_table <- function(predictions, class_labs, actual_class) {
  # find the 
  pred_class <- max.col(predictions)
  pred_class <- factor(class_labels[pred_class], levels = class_labels)
  tab <- table(actual_class, pred_class)
  # tab2 <- round(tab/rowSums(tab)*100, 2)
  # plot(tab2, las=2)
  print(kable(cbind(tab, class_acc=round(diag(tab)/rowSums(tab),3)), align = 'c'))
  cat("Overall Test Classification Accuracy: ", sum(diag(tab))/sum(tab))
}

# to add a smooth line to points
smooth_line <-function(y) {
  x <- 1:length(y)
  out <-predict(loess(y~x) )
  return(out)
}

# set labels
class_labels <-c("bathroom","bedroom","children_room",
                 "closet","corridor", "dining_room",
                 "garage","kitchen","living_room","stairs")

train_dir <- "Data/data_indoor/train"
validation_dir <- "Data/data_indoor/validation"
test_dir <- "Data/data_indoor/test"

# set our data augmentation generator
data_augment <-image_data_generator(
  rescale=1/255,
  zoom_range=0.2,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  horizontal_flip=TRUE,
  fill_mode="nearest"
)

validation_datagen <- image_data_generator(rescale=1/255)
test_datagen <- image_data_generator(rescale=1/255)

# train data generator with data augmentation
train_generator <-flow_images_from_directory(
  train_dir,
  data_augment,
  target_size=c(128, 128),
  batch_size=128,
  class_mode="categorical"
)

validation_generator <-flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size=c(128, 128),
  batch_size=128,
  class_mode="categorical"
)

test_generator <-flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size=c(128, 128),
  batch_size=128,
  class_mode="categorical",
  shuffle = FALSE
)

# generate vector of classes for the test data set
act_class <- factor(class_labels[(test_generator$classes+1)], levels = class_labels)

## 2. Deploy at least 3 different CNNs characterized by different configurations, 
## hyperparameters, and training settings (kernel size, filter size, pooling size, 
## regularization, etc.). 
## Motivate clearly the choices made in relation to the settings, configurations, 
## and hyperparameteres used to define the different CNNs.


# Model 1 ---------------------------------
model1 <- keras_model_sequential() %>%
  #
  # convolutional layers
  layer_conv_2d(filters=32, kernel_size=c(3,3), padding = "same", activation="relu", input_shape=c(128,128,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  #
  # fully connected layers
  layer_flatten() %>%
  layer_dense(units=256, activation="relu") %>%
  layer_dense(units=10, activation="softmax") %>%
  #
  # compile
  compile(
    loss="categorical_crossentropy",
    metrics="accuracy",
    optimizer =optimizer_adam()
  )

#NOTE: this will take time!
fit1 <- model1 %>% fit_generator(
  train_generator,
  steps_per_epoch=ceiling(train_generator$n/train_generator$batch_size),
  epochs=50,
  validation_data=validation_generator,
  validation_steps=ceiling(validation_generator$n/validation_generator$batch_size),
  verbose=1
)

# Evaluate/predict classes for test data set
model1 %>% evaluate_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))
preds1 <- model1 %>% predict_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))

# ---------------------------------

# Model 2 ---------------------------------
model2 <- keras_model_sequential() %>%
  #
  # convolutional layers
  layer_conv_2d(filters=32, kernel_size=c(3,3), padding = "same", activation="relu", input_shape=c(128,128,3)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_conv_2d(filters=256, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  #
  # fully connected layers
  layer_flatten() %>%
  layer_dense(units=256, activation="relu") %>%
  layer_dense(units=128, activation="relu") %>%
  layer_dense(units=10, activation="softmax") %>%
  #
  # compile
  compile(
    loss="categorical_crossentropy",
    metrics="accuracy",
    optimizer =optimizer_adam()
  )

#NOTE: this will take time!
fit2 <- model2 %>% fit_generator(
  train_generator,
  steps_per_epoch=ceiling(train_generator$n/train_generator$batch_size),
  epochs=50,
  validation_data=validation_generator,
  validation_steps=ceiling(validation_generator$n/validation_generator$batch_size),
  verbose=1
)

# Evaluate/predict classes for test data set
model2 %>% evaluate_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))
preds2 <- model2 %>% predict_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))

# ---------------------------------

# Model 3 ---------------------------------
model3 <- keras_model_sequential() %>%
  #
  # convolutional layers
  layer_conv_2d(filters=32, kernel_size=c(3,3), padding = "same", activation="relu", input_shape=c(128,128,3)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.1) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_conv_2d(filters=256, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.1) %>%
  #
  # fully connected layers
  layer_flatten() %>%
  layer_dense(units=256, activation="relu") %>%
  layer_dropout(0.4) %>%
  layer_dense(units=128, activation="relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units=10, activation="softmax") %>%
  #
  # compile
  compile(
    loss="categorical_crossentropy",
    metrics="accuracy",
    optimizer =optimizer_adam()
  )

#NOTE: this will take time!
fit3 <- model3 %>% fit_generator(
  train_generator,
  steps_per_epoch=ceiling(train_generator$n/train_generator$batch_size),
  epochs=50,
  validation_data=validation_generator,
  validation_steps=ceiling(validation_generator$n/validation_generator$batch_size),
  verbose=1
)

# Evaluate/predict classes for test data set
model3 %>% evaluate_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))
preds3 <- model3 %>% predict_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))

# ---------------------------------

# Model 4 ---------------------------------
model4 <- keras_model_sequential() %>%
  #
  # convolutional layers
  layer_conv_2d(filters=32, kernel_size=c(3,3), padding = "same", activation="relu", input_shape=c(128,128,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), padding = "same", activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  #
  # fully connected layers
  layer_flatten() %>%
  layer_batch_normalization() %>%
  layer_dense(units=128, activation="relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units=10, activation="softmax") %>%
  #
  # compile
  compile(
    loss="categorical_crossentropy",
    metrics="accuracy",
    optimizer =optimizer_adam(lr=0.0005)
  )

#NOTE: this will take time!
fit4 <- model4 %>% fit_generator(
  train_generator,
  steps_per_epoch=ceiling(train_generator$n/train_generator$batch_size),
  epochs=50,
  validation_data=validation_generator,
  validation_steps=ceiling(validation_generator$n/validation_generator$batch_size),
  verbose=1
)

# Evaluate/predict classes for test data set
model4 %>% evaluate_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))
preds4 <- model4 %>% predict_generator(test_generator, steps = ceiling(test_generator$n/test_generator$batch_size))

# ---------------------------------

# Classification table and accuracy for test set predictions --------
class_table(preds1, class_labels, act_class)
class_table(preds2, class_labels, act_class)
class_table(preds3, class_labels, act_class)
class_table(preds4, class_labels, act_class)
# ---------------------------------

# Plot accuracy for training and validation data --------
accuracy1_4 <- cbind(fit1$metrics$accuracy, fit1$metrics$val_accuracy,
                     fit2$metrics$accuracy, fit2$metrics$val_accuracy,
                     fit3$metrics$accuracy, fit3$metrics$val_accuracy,
                     fit4$metrics$accuracy, fit4$metrics$val_accuracy)

# accuracy1_4 <- cbind(fit1$metrics$accuracy[1:50], fit1$metrics$val_accuracy[1:50],
#                      fit2$metrics$accuracy[1:50], fit2$metrics$val_accuracy[1:50],
#                      fit3$metrics$accuracy[1:50], fit3$metrics$val_accuracy[1:50],
#                      fit4$metrics$accuracy[1:50], fit4$metrics$val_accuracy[1:50])

matplot(accuracy1_4, pch=20, col = adjustcolor(rep(c(1,3,6,7), each=2), 0.2), 
        main="Training and Validation Classification Accuracy vs Epochs", 
        ylab="Classification Accuracy", xlab="Epoch", cex=0.5)
matlines(apply(accuracy1_4, 2, smooth_line), lty=rep(c(1,2), 4), 
         col=adjustcolor(rep(c(1,3,6,7), each=2), 0.7), lwd=2)
legend("topleft", c("Model1 Train", "Model1 Val", "Model2 Train", "Model2 Val",
                    "Model3 Train", "Model3 Val", "Model4 Train", "Model4 Val"),
       lty = rep(c(1,2),2), col=rep(c(1,3,6,7), each=2), cex=0.6, bty="n")

# matplot(accuracy1_4, pch=20, col = adjustcolor(1:8, 0.2), 
#         main="Training and Validation Classification Accuracy vs Epochs", 
#         ylab="Classification Accuracy", xlab="Epoch")
# matlines(apply(accuracy1_4, 2, smooth_line), lty=rep(c(1,2), 4), col=1:8)
# legend("topleft", c("Model1 Train", "Model1 Val", "Model2 Train", "Model2 Val",
#                     "Model3 Train", "Model3 Val", "Model4 Train", "Model4 Val"),
#        lty = rep(c(1,2),2), col=1:8, cex=0.6, bty="n")
# ---------------------------------

# Plot loss for training and validation data --------
loss1_4 <- cbind(fit1$metrics$loss, fit1$metrics$val_loss,
                     fit2$metrics$loss, fit2$metrics$val_loss,
                     fit3$metrics$loss, fit3$metrics$val_loss,
                     fit4$metrics$loss, fit4$metrics$val_loss)

matplot(loss1_4, pch=20, col = adjustcolor(rep(c(1,3,6,7), each=2), 0.2), 
        main="Training and Validation Loss vs Epochs", 
        ylab="Loss", xlab="Epoch", cex=0.5, ylim=c(0,3))
matlines(apply(loss1_4, 2, smooth_line), lty=rep(c(1,2), 4), 
         col=adjustcolor(rep(c(1,3,6,7), each=2), 0.7), lwd=2)
legend("topleft", c("Model1 Train", "Model1 Val", "Model2 Train", "Model2 Val",
                    "Model3 Train", "Model3 Val", "Model4 Train", "Model4 Val"),
       lty = rep(c(1,2),2), col=rep(c(1,3,6,7), each=2), cex=0.6, bty="n")

# ---------------------------------













#######################################################
# Code from Lab 8
#######################################################

# to add a smooth line to points
smooth_line <-function(y) {
  x <-1:length(y)
  out <-predict(loess(y~x) )
  return(out)
}

# check learning curves
out <-cbind(fit$metrics$accuracy,
            fit$metrics$val_accuracy,
            fit$metrics$loss,
            fit$metrics$val_loss)

cols <-c("black","dodgerblue3")
par(mfrow =c(1,2))

# accuracy
matplot(out[,1:2],pch =19,ylab ="Accuracy",xlab ="Epochs",
        col =adjustcolor(cols,0.3),log ="y")
matlines(apply(out[,1:2],2, smooth_line),lty =1,col =cols,lwd =2)
legend("bottomright",legend =c("Training","Validation"),
       fill =cols,bty ="n")
#
# loss
matplot(out[,3:4],pch =19,ylab ="Loss",xlab ="Epochs",
        col =adjustcolor(cols,0.3))
matlines(apply(out[,3:4],2, smooth_line),lty =1,col =cols,lwd =2)
legend("topright",legend =c("Training","Validation"),
       fill =cols,bty ="n")










## 2 Data augmentation

# set our data augmentation generator
data_augment <-image_data_generator(
  rescale =1/255,
  rotation_range =40,
  width_shift_range =0.2,
  height_shift_range =0.2,
  shear_range =0.2,
  zoom_range =0.2,
  horizontal_flip =TRUE,
  fill_mode ="nearest"
)

# plot a couple of examples
par(mfrow =c(2,4),mar =rep(0.5,4))

# cute cat
img_array <-image_to_array(
  image_load("data_cats_dogs_small/train/cats/cat.10763.jpg",target_size =c(150,150))
)
img_array <-array_reshape(img_array,c(1,150,150,3))
augmentation_generator <-flow_images_from_data(
  img_array,
  generator =data_augment,
  batch_size =1
)
for(i in 1:4) {
  batch <-generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}

# cute dog
img_array <-image_to_array(
  image_load("data_cats_dogs_small/train/dogs/dog.9743.jpg",target_size =c(150,150))
)
img_array <-array_reshape(img_array,c(1,150,150,3))
augmentation_generator <-flow_images_from_data(
  img_array,
  generator =data_augment,
  batch_size =1
)
for(i in 1:4) {
  batch <-generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}


# deploy model
model_augment <-keras_model_sequential() %>%
  #
  # convolutional layers
  layer_conv_2d(filters =32,kernel_size =c(3,3),activation ="relu",
                input_shape =c(150,150,3)) %>%
  layer_max_pooling_2d(pool_size =c(2,2))%>%
  layer_conv_2d(filters =64,kernel_size =c(3,3),activation ="relu") %>%
  layer_max_pooling_2d(pool_size =c(2,2)) %>%
  layer_conv_2d(filters =128,kernel_size =c(3,3),activation ="relu") %>%
  layer_max_pooling_2d(pool_size =c(2,2)) %>%
  layer_conv_2d(filters =128,kernel_size =c(3,3),activation ="relu") %>%
  layer_max_pooling_2d(pool_size =c(2,2)) %>%
  #
  # fully connected layers
  layer_flatten() %>%
  layer_dense(units =512,activation ="relu") %>%
  layer_dense(units =1,activation ="sigmoid") %>%
  #
  # compile
  compile(
    loss ="binary_crossentropy",
    metrics ="accuracy",
    optimizer =optimizer_rmsprop(lr =0.0001)
    )

# train data generator with data augmentation
train_generator <-flow_images_from_directory(
  train_dir,
  data_augment,
  target_size =c(150,150),
  batch_size =20,
  class_mode ="binary"
  )

# train with data augmentation
#NOTE: this will take time!
fit_augment <-model_augment %>% fit_generator(
  train_generator,
  steps_per_epoch =100,
  epochs =30,
  validation_data =validation_generator,
  validation_steps =50
  )


# check accuracy learning curve
out_augment <-cbind(out[,1:2],
                    fit_augment$metrics$accuracy,
                    fit_augment$metrics$val_accuracy,
                    out[,3:4],
                    fit_augment$metrics$loss,
                    fit_augment$metrics$val_loss)

cols <-c("black","dodgerblue3","darkorchid4","magenta")
par(mfrow =c(1,2))
#
# accuracy
matplot(out_augment[,1:4],
        pch =19,ylab ="Accuracy",xlab ="Epochs",
        col =adjustcolor(cols,0.3),
        log ="y")
matlines(apply(out_augment[,1:4],2, smooth_line),lty =1,col =cols,lwd =2)
legend("bottomright",legend =c("Training","Valid","Aug_Training","Aug_Valid"),
       fill =cols,bty ="n")
#
# loss
matplot(out_augment[,5:8],pch =19,ylab ="Loss",xlab ="Epochs",
        col =adjustcolor(cols,0.3))
matlines(apply(out_augment[,5:8],2, smooth_line),lty =1,col =cols,lwd =2)
legend("topright",legend =c("Training","Valid","Aug_Training","Aug_Valid"),
       fill =cols,bty ="n")

model_augment %>% evaluate_generator(test_generator, steps = 50)
preds_aug <- model_augment %>% predict_generator(test_generator, steps = 50)

tab2 <- table(test_generator$classes, ifelse(preds_aug>0.5, 1,0))
tab2
sum(diag(tab2))/sum(tab2)






# Model 2 ---------------------------------
model2 <- keras_model_sequential() %>%
  #
  # convolutional layers
  layer_conv_2d(filters=32, kernel_size=c(3,3), activation="relu",
                input_shape=c(128,128,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  #
  # fully connected layers
  layer_flatten() %>%
  layer_batch_normalization() %>%
  # layer_dense(units=512, activation="relu") %>%
  layer_dense(units=64, activation="relu", kernel_regularizer=regularizer_l2(0.1)) %>%
  layer_dropout(0.1) %>%
  layer_dense(units=10, activation="softmax") %>%
  #
  # compile
  compile(
    loss="categorical_crossentropy",
    metrics="accuracy",
    # optimizer=optimizer_rmsprop(lr=0.0001)
    optimizer =optimizer_adam()
  )
