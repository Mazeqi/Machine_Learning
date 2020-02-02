class cs1():


    def copy(self,model,copymodel):
         while model:
            dell=model.pop()
            copymodel.append(dell)


    def printcopy(self,copymodel):
        for mod in copymodel:
            print(mod)



