import sys
sys.path.insert(0,"python")
import caffe
 
model="./deploy.prototxt"
 
def main():
    net=caffe.Net(model,caffe.TEST)
    this_layer_params=0
    flops=0
    blobs=net.blobs
    all_layer_name = []
    for key in blobs:             #feature map     
        all_layer_name.append(key)    #all layers name 
   # print(layer_name)


    layer_data_name = []    #params layers name
    layer_data = []         #params layers data (weights and bias)
    for item in net.params.items():   #params layers
        name1,layer1 = item
        layer_data_name.append(name1)   #params layers name
        layer_data.append(layer1)       #params layers data (weights and bias)
   # print(layer_data)
   # print(layer_data_name)
    print("layers_name".ljust(25)+"kernel_size".ljust(25)+"output_shape".ljust(25)+"param".ljust(25)+"flops".ljust(25))
    
    for i in range(len(layer_data_name)):
        #print(layer_data_name[i][-2:])
        if layer_data_name[i][-2:]=='bn' or layer_data_name[i][-2:] == 'le':
            #print('continue')
            continue
        if i == 0:
            try:
                
                print(all_layer_name[i].ljust(25) +"----".ljust(25)+ str(blobs[all_layer_name[i]].data.shape).ljust(25)+"----".ljust(25) + "----".ljust(25))
                weights_params = layer_data[i][0].count
                bias_params = layer_data[i][1].count
                #print(weights_params)
                this_layer_param = weights_params + bias_params
                #print(this_layer_param)
                #this_layer_param = layer_data[i][0].data
                output_feature_shape = blobs[layer_data_name[i]]
                flop = this_layer_param * output_feature_shape.width * output_feature_shape.height
                print(layer_data_name[i].ljust(25) + str(layer_data[i][0].data.shape).ljust(25)+str(blobs[layer_data_name[i]].data.shape).ljust(25)+str(this_layer_param).ljust(25)+str(flop).ljust(25))
                this_layer_params+=this_layer_param
                flops+=flop
                
            except:  #no bias 
                weights_params1 = layer_data[i][0].count
                #bias_params1 = layer_data[i][1].count
                this_layer_param1 = weights_params1# + bias_params1
                output_feature_shape1 = blobs[layer_data_name[i]]
                flop1 = this_layer_param1 * output_feature_shape1.width * output_feature_shape1.height
                print(layer_data_name[i].ljust(25) + str(layer_data[i][0].data.shape).ljust(25)+str(blobs[layer_data_name[i]].data.shape).ljust(25)+str(this_layer_param1).ljust(25)+ str(flop1).ljust(25))
                this_layer_params+=this_layer_param1
                flops+=flop1
        if i!=0:
            try:#for tiny ssd mobilenet ssd ->bn / scale layers
                weights_params1 = layer_data[i][0].count
                bias_params1 = layer_data[i][1].count
                this_layer_param1 = weights_params1 + bias_params1
                output_feature_shape1 = blobs[layer_data_name[i]]
                flop1 = this_layer_param1 * output_feature_shape1.width * output_feature_shape1.height
                print(layer_data_name[i].ljust(25)+ str(layer_data[i][0].data.shape).ljust(25)+str(blobs[layer_data_name[i]].data.shape).ljust(25)+str(this_layer_param1).ljust(25)+ str(flop1).ljust(25))
                this_layer_params+=this_layer_param1
                flops+=flop1
            except:  #no bias 
                weights_params1 = layer_data[i][0].count
                #bias_params1 = layer_data[i][1].count
                this_layer_param1 = weights_params1# + bias_params1
                output_feature_shape1 = blobs[layer_data_name[i]]
                flop1 = this_layer_param1 * output_feature_shape1.width * output_feature_shape1.height
                try:
                    print(layer_data_name[i].ljust(25) + str(layer_data[i][0].data.shape).ljust(25)+str(blobs[layer_data_name[i]].data.shape).ljust(25)+str(this_layer_param1).ljust(25)+ str(flop1).ljust(25))
                except:
                        try:
                            print(layer_data_name[i].ljust(25) + str(layer_data[i][0].data.shape).ljust(25)+str(blobs[layer_data_name[i]].data.shape).ljust(25)+str(this_layer_param1).ljust(25)+ str(flop1).ljust(25))
                        except:
                            print(layer_data_name[i].ljust(25) + str(layer_data[i][0].data.shape).ljust(25)+str(blobs[layer_data_name[i]].data.shape).ljust(25)+str(this_layer_param1).ljust(25)+ str(flop1).ljust(25))
                this_layer_params+=this_layer_param1
                flops+=flop1
    print("total params",this_layer_params)
    print("FLOPs:",flops)            



    
'''    
    for item in net.params.items():
        name,layer=item
        print(name)
        print(layer[0].data.shape)
        print(blobs[name].data.shape)
        c1=layer[0].count
        c2=layer[1].count
        b=blobs[name]
        param=c1+c2
        flop=param*b.width*b.height
        print(name+" "+str(param)+" "+str(flop))
        params+=param
        flops+=flop
    print("total params",params)
    print("FLOPs:",flops)
'''
if __name__ == '__main__':
    main()
