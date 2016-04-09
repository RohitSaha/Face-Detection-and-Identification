import cv2
import numpy as np
import pygame

unroll = np.zeros((1, 62500))
unroll = unroll[1:, :]
training_image = np.zeros((1, 62500))
output_array = np.zeros((1, 4), 'float')
true_label = np.zeros((4, 4), 'float')
for i in range(4):
    true_label[i, i] = 1

pygame.init()
disp = pygame.display.set_mode((50, 50))
disp.fill((0, 0, 255))
frames = 0
cap = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier("./haarcascade_frontalface_alt_tree.xml")

storing_data = True
take_picture1 = False
take_picture2 = False
take_picture3 = False
take_picture4 = False

while(cap.isOpened and storing_data):
    detect = False
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray)
    #gray = cv2.flip(frame, 1)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), 150]
    result, imgencode = cv2.imencode('.jpg', gray, encode)
    data = np.array(imgencode)
    #LOAD_IMAGE_GRAYSCALE converts the image iinto a 2-D matrix of grayscale.
    decimg = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #test = decimg

    for (x,y,w,h) in faces:
        detect = True
        cv2.rectangle(frame, (x,y+10),(x+w,y+h+20),(255,0,0))
        test = decimg[y+10:y+h+20,x:x+w]
        # r = 112 / test.shape[1]
        dim = (250 , 250)
        test = cv2.resize(test, dim, interpolation = cv2.INTER_AREA)
        unroll = test.reshape(1, 62500).astype(np.float32)
        try:
            cv2.imshow("test",test)
        except:
            print "Error"
        #test = np.asarray(test)
        #unroll = test.reshape(1, 62500).astype(np.float32)
    cv2.imshow("bla",frame)

    if(cv2.waitKey(30)==27&0xff):
        break
    if detect == True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key_input = pygame.key.get_pressed()
                if key_input[pygame.K_r]:
                    take_picture1 = True
                    print "Taking pictures of 1st person...."
                elif key_input[pygame.K_t]:
                    take_picture2 = True
                    print "Taking pictures of 2nd person...."
                elif key_input[pygame.K_y]:
                    take_picture3 = True
                    print "Taking pictures of 3rd person...."
                elif key_input[pygame.K_u]:
                    take_picture4 = True
                    print "Taking pictures of 4th person...."
                elif key_input[pygame.K_SPACE]:
                    print "Stopped taking pictures......"
                    take_picture1 = False
                    take_picture2 = False
                    take_picture3 = False
                    take_picture4 = False
                    frames = 0
                elif key_input[pygame.K_q]:
                    print "All data collected"
                    storing_data = False
                    pygame.display.quit()
                    break
        if(take_picture1 == True):
            frames += 1
            print "Frame number : ", frames
            training_image = np.vstack((training_image, unroll))
            output_array = np.vstack((output_array, true_label[0]))
            cv2.imwrite('training_images/rohit/training_image%d.jpg'%frames, test)
        elif(take_picture2 == True):
            frames += 1
            print "Frame number : ", frames
            training_image = np.vstack((training_image, unroll))
            output_array = np.vstack((output_array, true_label[1]))
            cv2.imwrite('training_images/ansh/training_image%d.jpg'%frames, test)
        elif(take_picture3 == True):
            frames += 1
            print "Frame number : ", frames
            training_image = np.vstack((training_image, unroll))
            output_array = np.vstack((output_array, true_label[2]))
            cv2.imwrite('training_images/alaap/training_image%d.jpg'%frames, test)
        elif(take_picture4 == True):
            frames += 1
            print "Frame number : ", frames
            training_image = np.vstack((training_image, unroll))
            output_array = np.vstack((output_array, true_label[3]))
            cv2.imwrite('training_images/aneek/training_image%d.jpg'%frames, test)

cv2.destroyAllWindows()
training_image = training_image[1:, :]
output_array = output_array[1:, :]

print training_image.shape
print output_array.shape

np.savez('training_data/training.npz', training_image_array = training_image, output_array = output_array)

