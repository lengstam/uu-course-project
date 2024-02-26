class Fish:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        # Pike seem dangerous
        self.members = ['Pike']


    def printMembers(self):
        print('Printing members of the Dangerous Fish class')
        for member in self.members:
            print('\t%s ' % member)