from Asbestos_Utils import name_list,get_file_extension,load_names,load_image_label,save_image_label
class HardExampleData(object):
    def __init__(self, path_hard_example_files,path_image_files):
        self.path_hard_example_files = path_hard_example_files
        self.hard_example_texts = name_list(path_hard_example_files, extension ='.txt')
        self.path_image_files = path_image_files
        self.image_names = name_list(path_image_files)
        self.image_extension = get_file_extension(self.image_names[-1])
        self.sizes = self.size_hard_example()
        print("Existing hard example files are\n")
        print(self.hard_example_texts)


    def import_hard_example(self, save_path, number_file):
        hard_example_names = load_names(self.path_hard_example_files, self.hard_example_texts[number_file].split('.')[0])
        self.hard_example_size = self.sizes[number_file]
        for name in hard_example_names:
            root_name,index1,index2,extension = self.separate_hard_example_name(name)
            hard_example_image,hard_example_label = self.get_hard_example(root_name,index1,index2,extension)
            save_image_label(save_path, name, hard_example_image, hard_example_label)


    def separate_hard_example_name(self,name):
        extension = get_file_extension(name)
        root_name = name.split('_H_')[0]
        indexes = name.split('_H_')[-1].split('.')[0]
        index1 = int(indexes.split('_')[0])
        index2 = int(indexes.split('_')[1])
        return root_name,index1,index2,extension


    def get_hard_example(self,root_name,index1,index2,extension):
        image_name = root_name+self.image_extension
        image,label = load_image_label(self.path_image_files, image_name)
        hard_example_image = image[index1:(index1 + self.hard_example_size),
                           index2:(index2 + self.hard_example_size)]
        hard_example_label = label[index1:(index1 + self.hard_example_size),
                           index2:(index2 + self.hard_example_size)]
        return hard_example_image,hard_example_label


    def size_hard_example(self):
        size = []
        for hard_example_set in self.hard_example_texts:
            size.append(int(hard_example_set.split(' S ')[-1].split(' ')[0]))
        return size