import numpy as np

class EigenfaceProjection():
    
    def __init__(self, original_image, n_components, eigenfaces):
        self.original_image = original_image
        self.n_components   = n_components
        self.eigenfaces     = eigenfaces
    
    def project_face(self):
        top_n_eigenfaces = self.eigenfaces[:,:self.n_components] 
        reconstruct      = top_n_eigenfaces @ top_n_eigenfaces.T @ self.original_image.flatten()
        self.projection  = reconstruct.reshape(self.original_image.shape)
    
    def add_components(self, n_components_to_add):
        if type(n_components_to_add).__name__ != 'int':
            raise ValueError('Only can add integer number of components.  Like 3 components.  Not 3.5 components.')
        
        flat_projection =  self.projection.flatten()
        flat_original   =  self.original_image.flatten()

        if 1 <= n_components_to_add < self.eigenfaces.shape[1]:
            #project OG image onto n more eigenfaces and add the projection
            new_eigenfaces  =  self.eigenfaces[:,self.n_components:(self.n_components+n_components_to_add)]
            flat_projection += np.uint8(new_eigenfaces @ new_eigenfaces.T @ flat_original)
            
        elif -self.eigenfaces.shape[1] < n_components_to_add <= -1:
            #project OG image onto last few eigenfaces and subtract the projection(gram schmidt)
            old_eigenfaces = self.eigenfaces[:,(self.n_components-n_components_to_add):self.n_components]
            flat_projection -= np.uint8(old_eigenfaces @ old_eigenfaces.T @ flat_original)

        self.projection =  flat_projection.reshape(self.original_image.shape)

    def set_filepath(self, filepath):
        self.filepath = filepath