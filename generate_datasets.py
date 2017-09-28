import numpy as np

### functions for superposing images
def coord_to_idx(x, y, length, width):
    ## turn x, y coordinate into an index of array
    ## we assume the origin to be the upper left corner of image
    ## we assume x corresponds to index along width
    return y*width+x
def idx_to_coord(idx, length, width):
    # return tuple (x, y)
    n_r = 0
    while (n_r+1) * width - 1 < idx:
        n_r += 1
    #n_r -= 1
    return idx-n_r*width, n_r
            
def embed_image(origin_x, origin_y, frame, image_to_embed):
    """
    Embed 2D image into a 2D frame
    starting from the location (origin_x, origin_y)
    return: frame embedded with image
    """
    image_length, image_width = image_to_embed.shape
    frame_length, frame_width = frame.shape
    #print('embedded image at location %d by %d'%(origin_x,origin_y))
    for i_x_idx, f_x_idx in enumerate(range(origin_x, origin_x+image_width)):
        for i_y_idx, f_y_idx in enumerate(range(origin_y, origin_y+image_length)):
            #print('index of image', i_y_idx, i_x_idx)
            #print('index of frame', f_y_idx, f_x_idx)
            frame[f_y_idx, f_x_idx] += image_to_embed[i_y_idx, i_x_idx]
    return frame

def calc_feasible_regions(origin_x, origin_y, image_length, 
                          image_width,frame_length, frame_width,
                          max_x_overlap, max_y_overlap):
    x_range = np.array(range(frame_width))
    y_range = np.array(range(frame_length))
    x_feasible = np.append(x_range[x_range>=origin_x+image_width-max_x_overlap],
                   x_range[x_range<=origin_x-image_width+max_x_overlap])
    #print('feasible x coordinates', list(x_feasible))
    y_feasible = np.append(y_range[y_range>=origin_y+image_length-max_y_overlap],
                   y_range[y_range<=origin_y-image_length+max_y_overlap])
    feasible_origins_idx = list()
    for x in x_feasible:
        for y in y_feasible:
            feasible_origins_idx.append(coord_to_idx(x, y, frame_length, frame_width))
    return feasible_origins_idx

def calc_init_feasible_set(f_length, f_width, image_length, image_width):
    feasible_li = list()
    for y in range(f_length-image_length):
        for x in range(f_width-image_width):
            feasible_li.append(coord_to_idx(x, y, f_length, f_width))
    return set(feasible_li)


def create_frame(f_length, f_width, image_source, max_n_images, max_overlap_x, max_overlap_y):
    ## create empty frame
    frame = np.zeros((f_length, f_width))
    n_images = np.random.randint(1, max_n_images+1)
    #print('number of images supposed to be',n_images)
    image_length, image_width = image_source[0].shape
    #feasible_origins = set(range((f_length-image_length+1)*(f_width-image_width+1)))
    feasible_origins = calc_init_feasible_set(f_length, f_width, image_length, image_width)
    count = 0
    for i in range(n_images):
        #print(feasible_origins)
        if not bool(feasible_origins):
            #print('number of images generated', count)   
            return frame, count
        count += 1
        ## randomly sample index from feasible_origins
        o_idx = np.random.choice(list(feasible_origins))
        o_x, o_y = idx_to_coord(o_idx, f_length, f_width)
        ## randomly draw an image from image_source and embed it
        image_to_embed = image_source[np.random.randint(len(image_source)),:,:]
        frame = embed_image(o_x, o_y, frame, image_to_embed)
        new_feasible_origins = calc_feasible_regions(o_x, o_y, image_length, image_width, 
                                                     f_length, f_width,
                                                     max_overlap_x, max_overlap_y)
        #print('new feasible', new_feasible_origins)
        feasible_origins = feasible_origins & set(new_feasible_origins)
    #print('number of images generated', count)    
    return (frame, count)

def create_dataset(f_length, f_width, image_source, max_n_images, max_overlap_x, max_overlap_y, n_samples):
    #print(create_frame(f_length, f_width, image_source, max_n_images, max_overlap_x, max_overlap_y))
    return [create_frame(f_length, f_width, image_source, max_n_images, max_overlap_x, max_overlap_y) \
                        for i in range(n_samples)]

#### create data generator for reconstruction
def mnist_clustering_generator(f_length, f_width, 
                               image_source, max_n_images, 
                               max_overlap_x, max_overlap_y, 
                               batch_size, total_n_samples):
    
    plain_label = np.zeros(max_n_images)
    n_epochs = 0
    data_collections = list()
    ind = 0
    ## process data to algorithm-friendly format
    #if model_type=='autoencoder':
    while n_epochs < total_n_samples/batch_size:
        ind = ind+1
        if ind < total_n_samples:
            # create new data-label pairs
            # get list of data-label tuples (list length equals batch_size)
            frame_label_list = create_dataset(f_length, f_width, image_source, 
                                          max_n_images, max_overlap_x, 
                                          max_overlap_y, batch_size)
            #print(frame_label_list)
            #frames, labels = zip(*frame_label_list)
            #print(frames)
            frames = list()
            labels = list()
            for i in range(len(frame_label_list)):
                frame, label = frame_label_list[i]
                frames.append(np.array(frame).flatten())
                labels.append(labels)
            frames = np.array(frames)
            labels = np.array(labels)
            data_collections.append((frames, labels))
        else:
            if ind == total_n_samples:
                n_epochs += 1
            ind = ind % total_n_samples
            frames, labels = data_collections[ind]
            #x.append(frame.flatten())
            #y_ = plain_label.copy()
            #y_[label-1] = 1
            #y.append(y_)
            #print(np.array(x).shape)
        #print(frames)
        yield (frames, frames)
        
####### create generator for predicting number of clusters
def mnist_clustering_generator_for_pred(f_length, f_width, 
                               image_source, max_n_images, 
                               max_overlap_x, max_overlap_y, 
                               batch_size, total_n_samples):
    
    plain_label = np.zeros(max_n_images)
    n_epochs = 0
    data_collections = list()
    ind = 0
    ## process data to algorithm-friendly format
    #if model_type=='autoencoder':
    while True:
        ind = ind+1
        label_template = np.zeros(max_n_images)
        if ind < total_n_samples:
            # create new data-label pairs
            # get list of data-label tuples (list length equals batch_size)
            frame_label_list = create_dataset(f_length, f_width, image_source, 
                                          max_n_images, max_overlap_x, 
                                          max_overlap_y, batch_size)
            #print(frame_label_list)
            #frames, labels = zip(*frame_label_list)
            #print(frames)
            frames = list()
            labels = list()
            for i in range(len(frame_label_list)):
                frame, label_ = frame_label_list[i]
                frames.append(np.array(frame).flatten())
                label = label_template.copy()
                label[label_-1] = 1
                labels.append(label)
            frames = np.array(frames)
            labels = np.array(labels)
            data_collections.append((frames, labels))
        else:
            if ind == total_n_samples:
                n_epochs += 1
            ind = ind % total_n_samples
            frames, labels = data_collections[ind]
            #x.append(frame.flatten())
            #y_ = plain_label.copy()
            #y_[label-1] = 1
            #y.append(y_)
            #print(np.array(x).shape)
        #print(frames)
        yield (frames, labels)

####### create dataset for predicting number of clusters
def mnist_clustering_dataset_for_pred(f_length, f_width, 
                               image_source, max_n_images, 
                               max_overlap_x, max_overlap_y, batch_size):
	label_templates = np.zeros(max_n_images)
	## process data to algorithm-friendly format
	#if model_type=='autoencoder':
	frame_label_list = create_dataset(f_length, f_width, image_source, max_n_images, max_overlap_x, max_overlap_y, batch_size)
	#print(frame_label_list)
	#frames, labels = zip(*frame_label_list)
	#print(frames)
	frames = list()
	labels = list()
	for i in range(len(frame_label_list)):
		frame, label_ = frame_label_list[i]
		frames.append(np.array(frame).flatten())
		label = label_templates.copy()
		label[label_-1] = 1
		labels.append(label)
	frames = np.array(frames)
	labels = np.array(labels)
        
	return frames, labels