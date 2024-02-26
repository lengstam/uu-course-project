class Birds:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        # Ducks are obviously dangerous
        self.members = ['Sparrow', 'Robin']


    def printMembers(self):
        print('Printing members of the Harmless Birds class')
        for member in self.members:
            print('\t%s ' % member)