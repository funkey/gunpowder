import logging
import random
import numpy as np

# imports for deformed slice
from skimage.draw import line
from scipy.ndimage.measurements import label
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_dilation

from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class DefectAugment(BatchFilter):

    def __init__(
            self,
            prob_missing=0.05,
            prob_low_contrast=0.05,
            prob_artifact=0.0,
            prob_deform=0.0,
            contrast_scale=0.1,
            artifact_source=None,
            deformation_strength=20,
            axis=0):
        '''Create a new DefectAugment node.

        Args

            prob_missing, prob_low_contrast, prob_artifact, prob_deform:

                Probabilities of having a missing section, low-contrast section,
                an artifact (see param 'artifact_source') or a deformed slice.
                The sum should not exceed 1.

        contrast_scale:

            By how much to scale the intensities for a low-contrast section.

        artifact_source:

            A gunpowder batch provider that delivers VolumeTypes.RAW and
            VolumeTypes.ALPHA_MASK, used if prob_artifact > 0.
                Strength of the deformation in slice.

        deformation_strength:

            Strength of the slice deformation.

        axis:

            Along which axis sections are cut.
        '''
        self.prob_missing = prob_missing
        self.prob_low_contrast = prob_low_contrast
        self.prob_artifact = prob_artifact
        self.prob_deform = prob_deform
        self.contrast_scale = contrast_scale
        self.artifact_source = artifact_source
        self.deformation_strength = deformation_strength
        self.axis = axis

    def setup(self):

        if self.artifact_source is not None:
            self.artifact_source.setup()

    def teardown(self):

        if self.artifact_source is not None:
            self.artifact_source.teardown()

    # send roi request to data-source upstream
    def prepare(self, request):

        # we prepare the augmentations, by determining which slices will be augmented by which method
        # already. If one of the slices is augmented with 'deform', we prepare these trafos already
        # and request a bigger roi from upstream

        prob_missing_threshold = self.prob_missing
        prob_low_contrast_threshold = prob_missing_threshold + self.prob_low_contrast
        prob_artifact_threshold = prob_low_contrast_threshold + self.prob_artifact
        prob_deform_slice = prob_artifact_threshold + self.prob_deform

        spec = request[VolumeTypes.RAW]
        roi = spec.roi
        logger.debug("downstream request ROI is %s" % roi)
        raw_voxel_size = self.spec[VolumeTypes.RAW].voxel_size

        # store the mapping slice to augmentation type in a dict
        self.slice_to_augmentation = {}
        # store the transformations for deform slice
        self.deform_slice_transformations = {}
        for c in range((roi / raw_voxel_size).get_shape()[self.axis]):
            r = random.random()

            if r < prob_missing_threshold:
                logger.debug("Zero-out " + str(c))
                self.slice_to_augmentation[c] = 'zero_out'

            elif r < prob_low_contrast_threshold:
                logger.debug("Lower contrast " + str(c))
                self.slice_to_augmentation[c] = 'lower_contrast'

            elif r < prob_artifact_threshold:
                logger.debug("Add artifact " + str(c))
                self.slice_to_augmentation[c] = 'artifact'

            elif r < prob_deform_slice:
                logger.debug("Add deformed slice " + str(c))
                self.slice_to_augmentation[c] = 'deformed_slice'
                # get the shape of a single slice
                slice_shape = (roi / raw_voxel_size).get_shape()
                slice_shape = slice_shape[:self.axis] + slice_shape[self.axis+1:]
                self.deform_slice_transformations[c] = self.__prepare_deform_slice(slice_shape)

        # prepare transformation and
        # request bigger upstream roi for deformed slice
        if 'deformed_slice' in self.slice_to_augmentation.values():

            # create roi sufficiently large to feed deformation
            # TODO ideally, we would reead this off of the transformations we already
            # created for the slice deformation, however this feels a bit over-engineered, because we know by how much we grow anyway...
            logger.debug("before growth: %s" % spec.roi)
            growth = Coordinate(
                tuple(0 if d == self.axis else raw_voxel_size[d] * self.deformation_strength for d in range(spec.roi.dims()))
            )
            logger.debug("growing request by %s" % str(growth))
            source_roi = roi.grow(growth, growth)

            # update request ROI to get all voxels necessary to perfrom
            # transformation
            spec.roi = source_roi
            logger.debug("upstream request roi is %s" % spec.roi)

    def process(self, batch, request):

        assert batch.get_total_roi().dims() == 3, "defectaugment works on 3d batches only"

        raw = batch.volumes[VolumeTypes.RAW]
        raw_voxel_size = self.spec[VolumeTypes.RAW].voxel_size

        for c, augmentation_type in self.slice_to_augmentation.items():

            section_selector = tuple(
                slice(None if d != self.axis else c, None if d != self.axis else c+1)
                for d in range(raw.spec.roi.dims())
            )

            if augmentation_type == 'zero_out':
                raw.data[section_selector] = 0

            elif augmentation_type == 'low_contrast':
                section = raw.data[section_selector]

                mean = section.mean()
                section -= mean
                section *= self.contrast_scale
                section += mean

                raw.data[section_selector] = section

            elif augmentation_type == 'artifact':

                section = raw.data[section_selector]

                alpha_voxel_size = self.artifact_source.spec[VolumeTypes.ALPHA_MASK].voxel_size

                assert raw_voxel_size == alpha_voxel_size, ("Can only alpha blend RAW with "
                                                            "ALPHA_MASK if both have the same "
                                                            "voxel size")

                artifact_request = BatchRequest()
                artifact_request.add(VolumeTypes.RAW, Coordinate(section.shape)*raw_voxel_size)
                artifact_request.add(VolumeTypes.ALPHA_MASK, Coordinate(section.shape)*alpha_voxel_size)
                logger.debug("Requesting artifact batch " + str(artifact_request))

                artifact_batch = self.artifact_source.request_batch(artifact_request)
                artifact_alpha = artifact_batch.volumes[VolumeTypes.ALPHA_MASK].data
                artifact_raw   = artifact_batch.volumes[VolumeTypes.RAW].data

                assert artifact_raw.dtype == section.dtype
                assert artifact_alpha.dtype == np.float32
                assert artifact_alpha.min() >= 0.0
                assert artifact_alpha.max() <= 1.0

                raw.data[section_selector] = section*(1.0 - artifact_alpha) + artifact_raw*artifact_alpha

            elif augmentation_type == 'deformed_slice':

                section = raw.data[section_selector].squeeze()

                # set interpolation to cubic, spec interploatable is true, else to 0
                interpolation = 3 if self.spec[VolumeTypes.RAW].interpolatable else 0

                # load the deformation fields that were prepared for this slice
                flow_x, flow_y, line_mask = self.deform_slice_transformations[c]

                # apply the deformation fields
                shape = section.shape
                section = map_coordinates(
                    section, (flow_y, flow_x), mode='constant', order=interpolation
                ).reshape(shape)

                # zero-out data below the line mask
                section[line_mask] = 0.

                raw.data[section_selector] = section

        # in case we needed to change the ROI due to a deformation augment,
        # restore original ROI and crop the volume data
        if 'deformed_slice' in self.slice_to_augmentation.values():
            old_roi = request[VolumeTypes.RAW].roi
            logger.debug("resetting roi to %s" % old_roi)
            crop = tuple(
                slice(None) if d == self.axis else slice(self.deformation_strength,-self.deformation_strength)
                for d in range(raw.spec.roi.dims())
            )
            raw.data = raw.data[crop]
            raw.spec.roi = old_roi

    def __prepare_deform_slice(self, slice_shape):

        # grow slice shape by 2 x deformation strength
        shape = (slice_shape[0] + 2*self.deformation_strength, slice_shape[1] + 2*self.deformation_strength)

        # randomly choose fixed x or fixed y with p = 1/2
        fixed_x = random.random() < .5
        if fixed_x:
            x0, y0 = 0, np.random.randint(1, shape[1] - 2)
            x1, y1 = shape[0] - 1, np.random.randint(1, shape[1] - 2)
        else:
            x0, y0 = np.random.randint(1, shape[0] - 2), 0
            x1, y1 = np.random.randint(1, shape[0] - 2), shape[1] - 1

        ## generate the mask of the line that should be blacked out
        line_mask = np.zeros(shape, dtype='bool')
        rr, cc = line(x0, y0, x1, y1)
        line_mask[rr, cc] = 1

        # generate vectorfield pointing towards the line to compress the image
        # first we get the unit vector representing the line
        line_vector = np.array([x1 - x0, y1 - y0], dtype='float32')
        line_vector /= np.linalg.norm(line_vector)
        # next, we generate the normal to the line
        normal_vector = np.zeros_like(line_vector)
        normal_vector[0] = - line_vector[1]
        normal_vector[1] = line_vector[0]

        # make meshgrid
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        # generate the vector field
        flow_x, flow_y = np.zeros(shape), np.zeros(shape)

        # find the 2 components where coordinates are bigger / smaller than the line
        # to apply normal vector in the correct direction
        components, n_components = label(np.logical_not(line_mask).view('uint8'))
        assert n_components == 2, "%i" % n_components
        neg_val = components[0, 0] if fixed_x else components[-1, -1]
        pos_val = components[-1, -1] if fixed_x else components[0, 0]

        flow_x[components == pos_val] = self.deformation_strength * normal_vector[1]
        flow_y[components == pos_val] = self.deformation_strength * normal_vector[0]
        flow_x[components == neg_val] = - self.deformation_strength * normal_vector[1]
        flow_y[components == neg_val] = - self.deformation_strength * normal_vector[0]

        # generate the flow fields
        flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)

        # dilate the line mask
        line_mask = binary_dilation(line_mask, iterations=10)

        return flow_x, flow_y, line_mask
