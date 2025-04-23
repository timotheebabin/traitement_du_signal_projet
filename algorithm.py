"""
Algorithm implementation
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------
# Create a fingerprint for an audio file based on a set of hashes
# ----------------------------------------------------------------------------


class Encoding:

   """
   Class implementing the procedure for creating a fingerprint 
   for the audio files

   The fingerprint is created through the following steps
   - compute the spectrogram of the audio signal
   - extract local maxima of the spectrogram
   - create hashes using these maxima

   """

   def __init__(self, nperseg=128,noverlap=32, min_distance=50, \
                 time_window=1., freq_window=1500):

      """
      Class constructor

      To Do
      -----

      Initialize in the constructor all the parameters required for
      creating the signature of the audio files. These parameters include for
      instance:
      - the window selected for computing the spectrogram
      - the size of the temporal window 
      - the size of the overlap between subsequent windows
      - etc.

      All these parameters should be kept as attributes of the class.
      """
      self.nperseg = nperseg
      self.noverlap = noverlap
      self.min_distance = min_distance
      self.time_window = time_window
      self.freq_window = freq_window
        

   def process(self, fs, s):

      """

      To Do
      -----

      This function takes as input a sampled signal s and the sampling
      frequency fs and returns the fingerprint (the hashcodes) of the signal.
      The fingerprint is created through the following steps
      - spectrogram computation
      - local maxima extraction
      - hashes creation

      Implement all these operations in this function. Keep as attributes of
      the class the spectrogram, the range of frequencies, the anchors, the 
      list of hashes, etc.

      Each hash can conveniently be represented by a Python dictionary 
      containing the time associated to its anchor (key: "t") and a numpy 
      array with the difference in time between the anchor and the target, 
      the frequency of the anchor and the frequency of the target 
      (key: "hash")


      Parameters
      ----------

      fs: int
         sampling frequency [Hz]
      s: numpy array
         sampled signal
      """

      self.fs = fs
      self.s = s

      f, t, Sxx = spectrogram(s, fs, nperseg=self.nperseg, \
                              noverlap=self.noverlap)

      self.S = Sxx
      self.f = f
      self.t = t

      coords = peak_local_max(Sxx, min_distance=self.min_distance, exclude_border=False)

      # Index to actual values
      constellation = np.array([(self.f[p[0]],self.t[p[1]]) for p in coords]) 
      self.anchors = constellation

      self.hashes = []
      # Sort constellation along the time axis
      constellation = constellation[np.argsort(constellation[:, 1])]

      for a in range(len(constellation)):
         for i in range(a+1, len(constellation)):
            # t_i - t_a > delta t
            if constellation[a][1] - constellation[i][1] > self.time_window:
                break
            
            # f_i - f_a < delta f
            if constellation[i][0] - constellation[a][0] < self.freq_window :
               hashcode = np.array([constellation[i][1]-constellation[a][1],
                                    constellation[a][0],constellation[i][0]])
               self.hashes.append({'t' : constellation[a][1], 'hash' : hashcode})

        
   def display_spectrogram(self, display_anchors=False):

      """
      Display the spectrogram of the audio signal

      Parameters
      ----------
      display_anchors: boolean
         when set equal to True, the anchors are displayed on the
         spectrogram
      """

      #   plt.pcolormesh(self.t, self.f/1e3, self.S, shading='gouraud')
      plt.pcolormesh(self.t, self.f/1e3, self.S)
      plt.xlabel('Time [s]')
      plt.ylabel('Frequency [kHz]')
      if(display_anchors):
         plt.scatter(self.anchors[:, 1], self.anchors[:, 0]/1e3)
      plt.show()



# ----------------------------------------------------------------------------
# Compares two set of hashes in order to determine if two audio files match
# ----------------------------------------------------------------------------

class Matching:

   """
   Compare the hashes from two audio files to determine if these
   files match

   Attributes
   ----------

   hashes1: list of dictionaries
      hashes extracted as fingerprints for the first audiofile. Each hash 
      is represented by a dictionary containing the time associated to
      its anchor (key: "t") and a numpy array with the difference in time
      between the anchor and the target, the frequency of the anchor and
      the frequency of the target (key: "hash")

   hashes2: list of dictionaries
      hashes extracted as fingerprint for the second audiofile. Each hash 
      is represented by a dictionary containing the time associated to
      its anchor (key: "t") and a numpy array with the difference in time
      between the anchor and the target, the frequency of the anchor and
      the frequency of the target (key: "hash")

   matching: numpy array
      absolute times of the hashes that match together

   offset: numpy array
      time offsets between the matches
   """

   def __init__(self, hashes1, hashes2):

      """
      Compare the hashes from two audio files to determine if these
      files match

      Parameters
      ----------

      hashes1: list of dictionaries
         hashes extracted as fingerprint for the first audiofile. Each hash 
         is represented by a dictionary containing the time associated to
         its anchor (key: "t") and a numpy array with the difference in time
         between the anchor and the target, the frequency of the anchor and
         the frequency of the target

      hashes2: list of dictionaries
         hashes extracted as fingerprint for the second audiofile. Each hash 
         is represented by a dictionary containing the time associated to
         its anchor (key: "t") and a numpy array with the difference in time
         between the anchor and the target, the frequency of the anchor and
         the frequency of the target
         
      """


      self.hashes1 = hashes1
      self.hashes2 = hashes2

      times = np.array([item['t'] for item in self.hashes1])
      hashcodes = np.array([item['hash'] for item in self.hashes1])

      # Establish matches
      self.matching = []
      for hc in self.hashes2:
            t = hc['t']
            h = hc['hash'][np.newaxis, :]
            dist = np.sum(np.abs(hashcodes - h), axis=1)
            mask = (dist < 1e-6)
            if (mask != 0).any():
               self.matching.append(np.array([times[mask][0], t]))
      self.matching = np.array(self.matching)

      # 1. creating an array "offset" containing the time offsets of the 
      #    hashcodes that match
      self.offset = np.array([self.matching[i][0] - self.matching[i][1] for i in range(len(self.matching))])
     
      # 2. implementing a criterion to decide whether or not both extracts
      #    match

      if len(self.offset) == 0:
         return False

      hist, _ = np.histogram(self.offset, bins=100, density=True)
      max_i = np.argmax(hist)
      val1 = hist[max_i]
      hist[max_i] = 0
      max_i2 = np.argmax(hist)
      val2 = hist[max_i2]

      self.is_match = val1 > 2*val2

            
   def display_scatterplot(self):

      """
      Display through a scatterplot the times associated to the hashes
      that match
      """

      plt.scatter(self.matching[:, 0], self.matching[:, 1])
      plt.show()


   def display_histogram(self):

      """
      Display the offset histogram
      """

      plt.hist(self.offset, bins=100, density=True)
      plt.xlabel('Offset (s)')
      plt.show()


