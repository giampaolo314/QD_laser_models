#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class that implements a range of quantum dot models with coherent fields. 
These are summarised in the supplementary material of 

| *Threshold Behavior in Quantum Dot Nanolasers: Effects of Inhomogeneous Broadening*  
| Giampaolo D’Alessandro, Gian Luca Lippi, and Francesco Papoff  
| Phys. Rev. A [submitted]  
| DOI:
    
The methods that compute the time derivatives of each model,
eq(t,z,p,q), reference the supplementary material of this article.

The class structure is 

+------------+---------+---------------+-----------------------+
| Class      | Model   | Identical QDs | Article to cite (DOI) | 
+============+=========+===============+=======================+
| CIM        | CIM     | Yes           | [1]                   |
+------------+---------+---------------+-----------------------+
| CIM\_d     | CIM     | No            | [3]                   |
+------------+---------+---------------+-----------------------+
| TPM        | TPM     | Yes           | [2]                   |
+------------+---------+---------------+-----------------------+
| TPM\_d     | TPM     | No            | [3]                   |
+------------+---------+---------------+-----------------------+
| TPM\_1F    | TPM\_1F | Yes           | [3]                   |
+------------+---------+---------------+-----------------------+
| TPM\_1F\_d | TPM\_1F | No            | [3]                   |
+------------+---------+---------------+-----------------------+

If you are using this code please reference the following papers, depending on 
the models you use:
    
[1] **Coherent-Incoherent Model (CIM)** - Carroll, M. A.;
D'Alessandro, G.; Lippi, G. L.; Oppo, G.-L. & Papoff, F. “Thermal,
Quantum Antibunching and Lasing Thresholds from Single Emitters to
Macroscopic Devices” Phys. Rev. Lett., 2021, **126**, 063902 - DOI:
`10.1103/PhysRevLett.126.063902
<https://doi.org/10.1103/PhysRevLett.126.063902>`_

[2] **Two-Particle Model (TPM)** - Papoff, F.; Carroll, M. A.; Lippi,
G. L.; Oppo, G.-L. & D’Alessandro, G. “Quantum correlations, mixed
states, and bistability at the onset of lasing” Physical Review A,
2025, **111**, l011501 - DOI: `10.1103/PhysRevA.111.L011501
<https://doi.org/10.1103/PhysRevA.111.L011501>`_

[3] **Two-Particle Model with negligible fermion-fermion interactions
(TPM\_1F) and models with different quantum dots (CIM\_d, TPM\_d and
TPM_1F\_d)** - D’Alessandro, G.; Lippi, G. L. & Papoff, F. “Threshold
Behavior in Quantum Dot Nanolasers: Effects of Inhomogeneous
Broadening” - Submitted to Phys. Rev. A

Version 1 - Mon Jul 27 2025

Contact: Francesco Papoff [f.papoff@strath.ac.uk]

Installation
------------

This class is self-contained. Just copy this file in a convenient folder and
follow the "Usage" examples below.

Usage
-----

Below are a few examples of how to use this class and its methods. Other examples, specific to each model, are at the end of the QD.py file.

.. code-block:: python

    import numpy as np
    import QD
    
    # In the following example replace QD_CIM with the model of your choice. Please
    # keep in m ind that QD_TPM_d is much much slower than any of the other models
    # and even integrating for a few time units may take hours.
    
    qd = QD.QD_CIM()
    
    # Check that the default parameters are suitable. These are defined by the 
    # method qd.get_default_parameters(). Some of them, listed in 
    # qd.p['allowed_keys'], are user-editable using the method qd.set_parameter(),
    # see example below.
    
    qd.p
    
    # Integrate for 100 time units from random initial conditions
    qd.integrate(np.linspace(0,100,201))
    
    # Plot the b field
    qd.plot_field('b')
    
    # Change the detuning to be 0.2γ (this erases the previously computed values)
    qd.set_parameter({'Delta_nu' : 0.2*qd.p['gamma']})
    
    # Integrate for 20 units of time with a fine sampling using default
    # initial conditions (random)
    qd.integrate(np.linspace(0,20,201))
    # Integrate for an additional 50 units of time with a finer sampling
    qd.integrate(np.linspace(0,50,1001))
    
    # Plot the physical fields
    qd.plot_field('b')
    qd.plot_field('Bb')
    qd.plot_field('Cc')
    qd.plot_field('bCv')
    qd.plot_field('Vc')
    
    # Integrate to equilibrium starting from the current final state. Note that
    # depending on the model and the initial conditions reaching equilibrium may
    # require integrating the model equations for a relatively large number of 
    # time units.  You can change the equilibrium condition by altering
    # qd.p['equilibrium_threhsold'].
    qd.set_init_cond(qd.z[:,-1])
    t_end, z_end = qd.equilibrium_value()
    print("Equilibrium has been reached at time: {0:.3f}".format(t_end))
    
    # Set the initial condition to the equilibrium value
    qd.set_init_cond(qd.z_eq)
    # Integrate for 20 units of time with a fine sampling using the 
    # equilibrium value as initial condition
    qd.integrate(np.linspace(0,20,201))
    # Plot the coherent field at equilibrium
    qd.plot_field('b')
    
    # Save the data to a pickle file with default file name
    qd.save_to_file('QD_example.pickle')


License
-------

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Disclaimer
----------

This software is provided "as is," without warranty of any kind, express or 
implied, including but not limited to the warranties of merchantability, 
fitness for a particular purpose, and non-infringement. In no event shall 
the authors or copyright holders be liable for any claim, damages, or 
other liability, whether in an action of contract, tort, or otherwise, 
arising from, out of, or in connection with the software or the use or 
other dealings in the software.

The user assumes all responsibility and risk for the use of this software. 
We make no representations or warranties about the suitability, reliability, 
availability, timeliness, and accuracy of the software for any purpose.

"""

# Numerical packages
import numpy as np
# Package to integrate the model ODEs.
import scipy.integrate
from scipy import sparse
# Package to write the output to a file (in binary format)
import pickle
# Get the date and time as seed for the random number generator
from datetime import datetime
# Support for type hints
import typing
# Plotting routines
import matplotlib.pyplot as plt
# To time the execution of the integration
import time
# To deal with regular expressions (for finding the number of fermionic fields)
import re

class QD:
    """
    A Python object to store and manage data related to the quantum dot models
    that include coherent and incoherent variables, e.g. CIM, TPM and TPM_1F.

    Attributes:
    -----------
    
        p : dictionary
            A dictionary to store the physical and numerical parameters.
            
        q : dictionary
            The indices that map the physical fields to the z array. They must
            have the form j_<field name> followed by an R or an I if the field
            is complex and the index refers to its real and imaginary part
            respectively. Examples are j_Bb or j_VcR. This dictionary contains
            also other auxiliary indices that are created by the objects that 
            need them for mapping purposes. These must not have this format,
            otherwise methods that identify the fields in the model will not 
            work correctly.
        
        t : numpy array
            A 1D NumPy array representing time in units of the inverse of the
            non-radiative decay rate, :math:`\gamma_{nr}`.
        
        z : numpy array
            A 2D NumPy array that stores the fields in real and imaginary
            part format (the mapping can be obtained by calling the method
            field_mapping). Each row corresponds to a different variable,
            while the columns correspond to the 't' values.
            
        z0 : numpy array
             A 1D n umpy array that contains the initial conditions for the
             field's model in real and imaginary form.
             
        z_eq : numpy array
               A 1D numpy array with the equilibrium values of the model fields
               in real and imaginary format.
            
    Methods
    -------
    
        eq : returns the time derivative of the fields.

        equilibrium_value : returns the equilibrium values of the fields
        
        field_mapping : returns the mapping between z and the physical fields
        
        field_values : returns the values of the physical fields
        
        get_default_parameters : returns the default parameter values
        
        integrate : integrates the model for a given integration time
        
        load_from_file : loads p, t and z from a pickle file.
        
        _number_of_fermion_ops : returns the number of pairs of fermion operators in a field
        
        plot_field : plots a user specified field as a function of time
        
        save_to_file : saves p, t and z to a pickle file.
        
        set_detuning : sets the detuning distribution for models with different quantum dots.
        
        set_init_cond : sets the initial condition of the model.
        
        set_parameter : sets or updates the values in the parameter dictionary 'p'.
        
        _SparsityJac : computes the sparsity matrix of the model's Jacobian.
    """

    def __init__(self):
        """
        Initializes the QD object with empty data structures.
        """
        self.q = {}  # Initialize an empty dictionary for the indices
        self.p = self.get_default_parameters()  # Initialize the parameters
        # List of keys that are allowed to be changed
        self.p['allowed_keys'] = ['gamma', 'gamma_c', 'gamma_nl', 'mu', 'r', 'N', \
                        'Delta_nu', 'g', 't_span_max', 'equilibrium_threshold']
        self.t = np.array([])  # Initialize an empty 1D NumPy array for t
        self.z = np.array([[]])  # Initialize an empty 2D NumPy array for z
        self.z0 = np.array([]) # Array with initial condition
        self.init_cond_set = False # No initial condition specified.
        
    def eq(self,t, z, p, q):
        """
        Returns the time derivative of the model. At this level this is
        just a place-holder method. It must be defined in each sub-class
        using the appropriate set of equations.
        
        Parameters
        ----------
        
        t : real
          The time value at which the derivative is computed.
          
        z : 1D numpy array
          The current values of the model variables in real and imaginary form.
          
        p : dictionary
          The model's parameters
          
        q : dictionary
          The indices that map the physical fields to the z array

        Returns
        -------
        
        zdot : 1D numpy array
          The time derivatives at time t of the model variables z.
          
        """

        z_dot = np.array([])

        return z_dot
 
    def equilibrium_value(self):
        """
        Function to integrate the Coherent-Incoherent model with the one
        particle approximation for Fermions and identical quantum dots
        (CIM) until equilibrium has been reached, i.e. until a set of time
        derivatives is smaller in modulus than a threshold specified by
        p['equilibrium_threshold']. The integration stops with an error
        message if the equilibrium is not reached by time p['t_span'].

        Usage
        -----

        my_qd_cim = QD_CIM()

        my_qd_cim.equilibrium_value()

        Integrates the equations with random initial conditions with the
        object parameters specified in the dictionary p and initial conditions
        my_qd_cim.z0. If these are not defined they are set to random.

        Returns
        -------
        
        t_end : real
          The final integration time. If smaller than the maximum integration
          time this value is the time to equilibrium.
          
        z_end : numpy array
          The final value of the model variables z stored in real and imaginary 
          part format. If the integration time is smaller than the maximum
          integration time these can be considered as the equilibrium values
          to numerical precision and they are also stored in the object 
          variable z_eq.
          
        Raises
        ------
        
        EquilibriumNotReachedError
          The integration has reached the maximum allocated integration time
          without reaching its equilibrium

        """

        # Check if the initial condition has been set and set it to random otherwise
        if not self.init_cond_set:
            self.set_init_cond('random')
            
        z_init = self.z0
        
        def derivative_modulus_event(t, u):
            """
            Event function for solve_ivp to stop integration.
            
            Returns
            -------

            event_value : real
              This is equal to math:`\max([|\dot{|b|}|/\max(|b|,0.1),|\dot{b^{\dagger}b}|/|b^{\dagger}b|,\dot{c^{\dagger}c}/|c^{\dagger}c|]) - \mathrm{p['equilibrium_threshold']}`.
              An event is detected when this value crosses zero. 
            """
            
            # Compute the time derivative of all the fields
            u_dot = self.eq(t,u,self.p,self.q)
            
            # Extract the time derivative of the coherent field amplitude. If the
            # amplitude is significantly different from zero we compute the time
            # derivative of |<b>|, otherwise we compute the time derivative of |<b>|^2.
            abs_b = np.sqrt(u[self.q['j_bR']]**2 + u[self.q['j_bI']]**2)
            abs_b_dot = np.abs(u[self.q['j_bR']]*u_dot[self.q['j_bR']] \
                               + u[self.q['j_bI']]*u_dot[self.q['j_bI']]) / np.max([abs_b,0.1])
            # Extract the time derivative of the population inversion
            abs_Cc_dot = np.abs(u_dot[self.q['j_Cc']]) / np.abs(u[self.q['j_Cc']])
            if self.p['Model'][-2:] == "_d":
                # Model with different quantum dots, abs_Cc_dot is a vector and 
                # we take its maximum.
                abs_Cc_dot = abs_Cc_dot.max()
                
            # Extract the time derivative of the number of photons amplitude.
            abs_Bb_dot = np.abs(u_dot[self.q['j_Bb']]) / np.abs(u[self.q['j_Bb']])
            
            return np.max([abs_b_dot, abs_Cc_dot, abs_Bb_dot]) - self.p['equilibrium_threshold']

        # Set event properties:
        # `terminal=True` means the integration stops when the event
        # occurs.  `direction=-1` means the event is detected when the
        # function crosses zero from positive to negative (i.e.,
        # `abs(dx/dt)` drops below `p`).
        derivative_modulus_event.terminal = True
        derivative_modulus_event.direction = -1

        # Compute the sparsity pattern
        JPattern = self._SparsityJac(self.eq, self.z0, self.p, self.q) 

        # Set the options for the integrator
        # scipy.integrate.solve_ivp uses a dictionary for options
        # The Jacobian sparsity pattern is passed via `jac_sparsity` argument.
        options = {
            'rtol': 1e-5,
            'atol': 1e-7,
            # The three methods below are all valid for stiff
            # problems. Uncomment the one you wish to use.
            # 'method': 'LSODA' # Does not use the sparsity matrix
            # 'method': 'Radau'
            'method': 'BDF'
        }
        if self.p['Model'] in {'CIM'}:
            options['method'] = 'LSODA'
        if options['method'] in {'Radau', 'BDF'}:
          options['jac_sparsity'] = JPattern
        
        # Integrate the equations
        # The `CIM_eq` function needs `p` as an argument.
        # `solve_ivp` allows passing additional arguments to the fun.
        # `fun(t, y, *args)`
        start = time.time()
        sol = scipy.integrate.solve_ivp(fun=lambda t, z_ode: self.eq(t, z_ode, self.p, self.q), 
                        t_span=[0, self.p['t_span_max']], 
                        y0=z_init, 
                        t_eval=None, 
                        events=derivative_modulus_event,
                        #dense_output=True # Allows for evaluating the solution at arbitrary points
                        **options)
        end = time.time()
        print("Integration time: {0:.3f} seconds".format(end - start))
        # Store the output value
        t_end = sol.t[-1]
        z_end = sol.y[:,-1]
        if sol.t[-1]>= self.p['t_span_max']:
            # Equilibirum has not been reached in the allocated time.
            print('Equilibrium not reached')
        else:
            self.z_eq = z_end
        return t_end, z_end

    def field_mapping(self):
        """
        Prints the mapping from the fields in real and imaginary parts as 
        stored in the numpy array z and the physical fields.

        Returns
        -------
        
        None.
        
        """
        
        print("In the following the Hermitian conjugate of a field is indicated")
        print("with a capital letter. For example, B is the hermitian conjugate")
        print("of b.\n")

        for key in self.q.keys():
            if key[1]=='_':
                # The key corresponds to a field and can be used in the mapping
                # Check if the key ends with 'R'
                if key.endswith('R'):
                    field = key[2:-1]
                    nf = int(np.floor(len(re.findall(r'[CVcv]',field))/2))
                    if isinstance (self.q[key], int):
                        # This applies to bosonic field for all models and to
                        # all fields for models with identical quantum dots
                        print("{0:s} = z[{1:d}] + j*z[{2:d}]".format(field, \
                                self.q[key],self.q["j_{0:s}I".format(field)]))
                    elif (nf==1):
                        print("{0:s}[0:{1:d}] = z[{2:d}:{3:d}] + j*z[{4:d}:{5:d}]".format(field, \
                            self.p['N']-1,self.q[key][0],self.q[key][-1],\
                            self.q["j_{0:s}I".format(field)][0],self.q["j_{0:s}I".format(field)][-1]))
                    elif (nf==2):
                        print("{0:s}[n,m] = z[{1:d}+{2:d}n+m] + j*z[{3:d}+{2:d}m+n]".format(field, \
                            self.q[key][0],self.p['N'],self.q["j_{0:s}I".format(field)][0]))
                    else:
                        pass
                    # Check if the key ends with 'I'
                elif key.endswith('I'):
                    # Ignore keys ending in 'I'
                    pass # Explicitly pass, though it's optional here as there's no other code
                    # If it ends in neither 'R' nor 'I'
                else:
                    field = key[2:]
                    nf = int(np.floor(len(re.findall(r'[CVcv]',field))/2))
                    if isinstance (self.q[key], int):
                        # This applies to bosonic field for all models and to
                        # all fields for models with identical quantum dots
                        print("{0:s} = z[{1:d}]".format(field,self.q[key]))
                    elif (nf==1):
                        print("{0:s}[0:{1:d}] = z[{2:d}:{3:d}]".format(field,self.p['N']-1,self.q[key][0],self.p['N']-1))
                    elif (nf==2):
                        print("{0:s}[n,m] = z[{1:d}+{2:d}n+m]".format(field, \
                            self.q[key][0],self.p['N']))
                    else:
                        pass
    
    def field_values(self, field_name: str, j_QD: typing.Union[np.ndarray, None] = None) -> typing.Union[np.ndarray, None]:
        """
        Returns the values of the physical fields from those stored in real 
        and imaginary format in the z array.

        Parameters
        ----------
        
            field_name : string
              The name of the field to retrieve. The naming convention is 
              that the Hermitian conjugate is represented by the capital 
              letter. For example, B is :math:`b^{\dagger}`.
              
            j_QD : numpy array or None (optional)
              For models with different quantum dots it can be:
                  
              a) For single fermion operators, e.g. Vc, a 1D numpy array 
                 of the indices of the quantum dots.  
                 
              b) For two fermion operators, e.g. CCcc, a 2D numpy array of
                 the indices of the two quantum dots that define the operator.
                 
              c) If None, the field values for all quantum dots or pairs of
                 quantum dots are returned.
                 

        Returns
        -------
        
            anonymous : np.ndarray
              The requested field(s) from the z array. If the 
              field_name is invalid or z is not appropriately shaped, the
              function returns None.
            
        """
        if self.z.size == 0 or self.z.ndim != 2:
            print("Error: 'z' array is empty or not 2-dimensional. Cannot retrieve fields.")
            return None
        if self.z.shape[0] != self.p['nf']:
            print(f"Error: 'z' array does not have {self.p['nf']} columns. Cannot retrieve fields.")
            return None
        
        # Create a dictionary of allowed fields and indicate which are real and 
        # which are complex
        allowed_fields = {}
        for key in self.q.keys():
            if key[1]=='_':
                # The key corresponds to a field and can be used in this function
                # Check if the key ends with 'R'
                if key.endswith('R'):
                    field = key[2:-1]
                    allowed_fields[field] = 'C'
                # Check if the key ends with 'I'
                elif key.endswith('I'):
                    # Ignore keys ending in 'I'
                    pass # Explicitly pass, though it's optional here as there's no other code
                # If it ends in neither 'R' nor 'I'
                else:
                    field = key[2:]
                    allowed_fields[field] = 'R'
        
        try:
            if j_QD is None: #not isinstance(j_QD, np.ndarray):
                # The index is not an np.array. We assume that it has not been
                # specified. This is either a model with identical quantum 
                # dots or all the fields are to be returned.
                if allowed_fields[field_name] == 'R':
                    return self.z[self.q["j_{:s}".format(field_name)]]
                else:
                    return self.z[self.q["j_{:s}R".format(field_name)]] \
                        + 1j*self.z[self.q["j_{:s}I".format(field_name)]]
            else:
                # Check that the range of j_Qd is compatible with j_QD being
                # a quantum dot index.
                if j_QD.min() < 0:
                    print("Error: the minimum value of j_QD cannot be negative")
                    return None
                if j_QD.max() > self.p['N']:
                    print("Error: the maximum value of j_QD must be smaller than {:d}.".format(self.p['N']))
                    return None
                # Determine the number of fermion operators
                nf = int(np.floor(len(re.findall(r'[CVcv]',field_name))/2))
                if nf==0:
                    # Boson operator, the index should have been None.
                    print("The value of the index when requesting the values " \
                          + "of a boson operator should be None or left empty.")
                    return None
                elif nf == 1:
                    # One fermion operator.
                    # Check that the index is a 1D numpy array
                    if j_QD.ndim != 1:
                        print("When requesting a single fermion operator " \
                              + "like {:s} you should use a 1D numpy".format(field_name) \
                              + " array for the second argument or use None.")
                        return None
                    if allowed_fields[field_name] == 'R':
                        return self.z[self.q["j_{:s}".format(field_name)][j_QD]]
                    else:
                        return self.z[self.q["j_{:s}R".format(field_name)][j_QD]] \
                               + 1j*self.z[self.q["j_{:s}I".format(field_name)][j_QD]]
                elif nf == 2:
                    # Two fermion operator.
                    # Check that the index is a 2D numpy array
                    if j_QD.ndim != 2:
                        print("When requesting a two fermion operator " \
                              + "like {:s} you should use a 2D numpy".format(field_name) \
                              + " array for the second argument or use None.")
                        return None
                    # Create the list of indices
                    k_QD = []
                    k_QD_I = []
                    for k in j_QD:
                        if k[0]==k[1]:
                            print("Skipped indices [{:d},{:d}].".format(k[0],k[1]))
                            print("The indices of two-fermion operators must be different.")
                        elif allowed_fields[field_name] == 'R':
                            k_QD.append(self.q["jm_{:s}".format(field_name)][k[0],k[1]])
                        else:
                            k_QD.append(self.q["jm_{:s}R".format(field_name)][k[0],k[1]])
                            k_QD_I.append(self.q["jm_{:s}I".format(field_name)][k[0],k[1]])
                    if allowed_fields[field_name] == 'R':
                        return self.z[k_QD]
                    else:
                        return self.z[k_QD] + 1j*self.z[k_QD_I]
                    
        except Exception:
            print(f"Error: Invalid field name '{field_name}'.") 
            print("Accepted values are")
            print(list(allowed_fields.keys()))
            # Return None if an error occurred or field_name was invalid
            return None
    
    def get_default_parameters(self) -> dict:
        """
        Returns the default parameter values

        Returns
        -------
        
        p : dictionary
          A dictionary with the default parameter values

        """
        # Define a dictionary to contain all the parameters
        p = {}
        # Parameter: the parameters have been normalised to the
        # non-radiative decay rate, :math:`\gamma_{nr}=10^9`.
        # Dephasing rate of the photon-assisted polarisation
        p['gamma'] = 1e4
        # Photon decay rate in the cavity
        p['gamma_c'] = 10
        # Spontaneous emission rate into non-lasing modes
        # :math:`\beta = \gamma_{l}/(\gamma_{l}+\gamma_{nl})`
        # which implies that :math:`\gamma_{nl} = \gamma_{l}(1-\beta)/\beta`
        # with :math:`\gamma_{l} = 0.968`.
        # See Carroll et al, PRL 126, 063902 (2021), page 3
        p['gamma_nl'] = 0      # :math:`\beta = 1.00` [nanolaser]
        # p['gamma_nl'] = 1400   # :math:`\beta = 7 \times 10^{-4}`
        # p['gamma_nl'] = 2.8e5  # :math:`\beta = 3.4 \times 10^{-6}` [macroscopic laser]
        # Non-radiative decay rate
        p['gamma_nr'] = 1
        # p.mu is a positive parameter that models the effect of the
        # phono dephasing on two-fermion correlations.  :math:`\mu=0`
        # corresponds to maximum effect.  Even a relatively small
        # value of mu, e.g. :math:`\mu=0.05`, reduces the effect of
        # the correlations substantially.
        p['mu'] = 0

        # Pump value for fixed pump simulations
        p['r'] = 2e5

        # Number of quantum dots
        p['N'] = 30
        
        # Detuning - This is model dependent. The detuning is Δν=ν-Δε, with ν the
        # field cavity frequency and Δε the quantum dot frequency (difference
        # between energy levels).
        p['Delta_nu'] = 0.0*p['gamma']
    
        # Maximum value of the light-matter coupling strength
        p['g'] = 70 
        
        # Maximum integration time for finding an equilibrium value
        p['t_span_max'] = 1e5
        # The integration stops if the modulus of the derivative of the coherent
        # field is smaller than p.equilibrium_threshold.
        p['equilibrium_threshold'] = 1.e-3
        
        return p

    def integrate(self, t_int: np.ndarray):
        """
        Function to integrate the Coherent-Incoherent model with the one
        particle approximation for Fermions and identical quantum dots
        (CIM) for a time specified in the call to the function.

        Usage
        -----

        my_qd_cim = QD_CIM()

        my_qd_cim.integrate(t_int)

        Integrates the equations with random initial conditions with the
        default object parameters. t_int is a numpy 1D array with first element
        0. It specify the output times. 
        
        For example: 
            
            my_qd_cim = QD_CIM()
            
            my_qd_cim.integrate(np.linspace(0,10,41))
            
        integrates the CIM equations for 10 time units with outputs at 0, 0.25,
        0.50, etc. starting from random initial conditions using the default
        parameters.
        
        A subsequent call to the function continues the integration from the 
        last value reached in the previous call. 
        
        For example:
            my_qd_cim = QD_CIM()
            
            my_qd_cim.integrate(np.linspace(0,10,41))
            
            my_qd_cim.integrate(np.linspace(0,10,21))
            
        first integrates the CIM equations for 10 time units with outputs at 0, 0.25,
        0.50, etc. starting from random initial conditions using the default
        parameters. It then continues the integration for another 10 time units 
        with outputs at 10.5, 11.0, 11.5 etc. 
            

        Parameters
        ----------
        
        t_int : numpy 1D array 
          The requested output times. Its first element must be 0. The array 
          must have at least two elements.

        Returns
        -------

        Nothing. The output is stored in the object variables t and z.

        """

        # Check the input argument
        if not isinstance(t_int, np.ndarray):
            print("The input argument must be a 1D numpy array, e.g. np.linspace(0,10,50).")
            return None
        if t_int.size < 2 or t_int.ndim != 1:
            print("Error: the input argument must be a 1D numpy array with at least two elements.")
            return None
        if any(np.diff(t_int)<0):
            print("Error: the elements of the input argument must be increasing.")
            return None
        if t_int[0]!=0:
            print("Error: the first element of the input argument must be 0.")
        
        # If a solution has already been computed then we adjust t_int so that
        # the simulation continues from the last computed time.
        if self.t.size != 0:
            t_int = self.t[-1] + t_int
            z_init = self.z[:,-1]
        else:
            # Check if the initial condition has been set and set it to random otherwise
            if not self.init_cond_set:
                self.set_init_cond('random')
            z_init = self.z0

        # Compute the sparsity pattern
        JPattern = self._SparsityJac(self.eq, self.z0, self.p, self.q) # Pass CIM_eq as the function handle

        # Set the options for the integrator
        # scipy.integrate.solve_ivp uses a dictionary for options
        # The Jacobian sparsity pattern is passed via `jac_sparsity` argument.
        options = {
            'rtol': 1e-5,
            'atol': 1e-7,
            # The three methods below are all valid for stiff problems. Uncomment
            # the one you wish to use.
            # 'method': 'LSODA' # Does not use the sparsity matrix
            # 'method': 'Radau'
            'method': 'BDF'
        }
        if self.p['Model'] in {'CIM'}:
            options['method'] = 'LSODA'
        if options['method'] in {'Radau', 'BDF'}:
          options['jac_sparsity'] = JPattern
        
        # Integrate the equations
        # The `CIM_eq` function needs `p` as an argument.
        # `solve_ivp` allows passing additional arguments to the fun.
        # `fun(t, y, *args)`
        start = time.time()
        sol = scipy.integrate.solve_ivp(fun=lambda t, z_ode: self.eq(t, z_ode, self.p, self.q), 
                        t_span=[t_int[0], t_int[-1]], 
                        y0=z_init, 
                        t_eval=t_int, 
                        **options)
        end = time.time()
        print("Integration time: {0:.3f} seconds".format(end - start))
        # Append the output to self.t and self.z
        if self.t.size==0:
            self.t = sol.t
            self.z = sol.y
        else:
            self.t = np.concatenate((self.t,sol.t[1:]))
            self.z = np.concatenate((self.z, sol.y[:,1:]),axis=1)
        
    def load_from_file(self, filename: str = "QD.pickle"):
        """
        Loads the parameter dictionary p, the time values 1D numpy array t, 
        and the model variable 2D numpy array z into the QD object from
        a pickle file.

        Parameters
        ----------
        
            filename : string
              The name of the file to load the data from. Defaults to 
              "QD.pickle" if not provided.
              
        Raises
        ------
        
            FileNotFoundError
              The file specified cannot be located.
              
            pickle.UnpicklingError
              The file cannot be unpacked correctly
        """
        try:
            with open(filename, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Assign loaded data to instance attributes
            self.p = loaded_data.get('p') 
            self.q = loaded_data.get('q') 
            self.t = loaded_data.get('t')
            self.z = loaded_data.get('z')
            # Set the final value of z to be the initial condition
            self.z0 = self.z[:,-1]
            self.init_cond_set = True # We have specified an initial condition.
            
            print(f"Data loaded successfully from {filename}")
            # print(f"  Loaded p: {self.p}")
            # print(f"  Loaded q: {self.q}")
            # print(f"  Loaded t shape: {self.t.shape}")
            # print(f"  Loaded z shape: {self.z.shape}")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except pickle.UnpicklingError as e:
            print(f"Error: Could not unpickle data from '{filename}'. It might be corrupted or not a valid pickle file. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading from '{filename}': {e}")
    
    def plot_field(self, field_name: str, j_QD: typing.Union[np.ndarray, None] = None):
        """
        Plots the field specified in the input string. In the case of complex
        fields it plots the real and imaginary parts and the modulus.

        Parameters
        ----------
        
        field_name : string
          The name of the field to retrieve. The value allowed depend on
          the model. For example, for the CIM they are "b", "Vc", "Cc", 
          "Bb" or "bCv". A capital letter indicates the hermitian conjugate
          of the corresponding field, e.g. B = :math:`b^{\dagger}`.
          
        j_QD : numpy array or None (optional)
          For models with different quantum dots it can be:
              
          a) For single fermion operators, e.g. Vc, a 1D numpy array 
             of the indices of the quantum dots.  
             
          b) For two fermion operators, e.g. CCcc, a 2D numpy array of
             the indices of the two quantum dots that define the operator.
             
          c) If None, the field values for all quantum dots or pairs of
             quantum dots are plotted.

        """
        
        f = self.field_values(field_name,j_QD)
        
        # Check that field_values has returned a field
        if f is None:
            return None
                
        # Write the field name in LaTeX format
        
        def replace_uppercase_with_dagger(input_string):
            """
            Replaces any instance of an uppercase letter in a string with its
            lowercase equivalent followed by :math:`^{\dagger}`.
            
            Args:
                input_string (str): The string to process.

            Returns:
                str: The modified string.
            """
            modified_string = [] # Use a list to build the string efficiently
            for char in input_string:
                if 'A' <= char <= 'Z': # Check if the character is an uppercase letter
                    modified_string.append(char.lower())
                    modified_string.append("^{\\dagger}") # Add the dagger string
                else:
                    modified_string.append(char) # Keep other characters as they are
            return "".join(modified_string) # Join the list into a single string
        
        field_latex = replace_uppercase_with_dagger(field_name)
        
        # Determine if we are plotting a real or a complex field
        is_f_complex = np.iscomplexobj(f)     
        
        # Plot the data
        xlabel = "$t \gamma_{nr}$"
        if is_f_complex:
            ylabel = r"$\langle {:s} \rangle$".format(field_latex)
            field_labels = [r"$\Re(\langle {0:s} \rangle)$".format(field_latex), \
                r"$\Im(\langle {0:s} \rangle)$".format(field_latex), \
                r"$|\langle {0:s} \rangle|$".format(field_latex)]
        else:
            ylabel = r"$\langle {:s} \rangle$".format(field_latex)
            field_labels = [ylabel]
        # Index of the fields to be plotted
        m = 0
             
        plt.figure(figsize=(10, 6)) # Create a new figure for the plot
        if j_QD is None:
            # We are plotting either a single field in a model with identical 
            # quantum dots or, in the case of different quantum dots, either
            # a bosonic field or all the fermionic fields with the given name.
            # In all these cases the labels have no indices.
            if is_f_complex:
                plt.plot(self.t, np.real(f), label=field_labels[0])
                plt.plot(self.t, np.imag(f), label=field_labels[1])
                plt.plot(self.t, np.abs(f), label=field_labels[2])
                plt.legend()
            else:
                plt.plot(self.t, f)
        elif j_QD.ndim == 1:
            # We are plotting a single fermion field
            if is_f_complex:
                for j in j_QD:
                    plt.plot(self.t, np.real(f[m,:]), label=field_labels[0]+"$_{{{:d}}}$".format(j))
                    plt.plot(self.t, np.imag(f[m,:]), label=field_labels[1]+"$_{{{:d}}}$".format(j))
                    plt.plot(self.t, np.abs(f[m,:]), label=field_labels[2]+"$_{{{:d}}}$".format(j))
                    m = m + 1
            else:
                for j in j_QD:
                    plt.plot(self.t, f[m,:], label=field_labels[0]+"$_{{{:d}}}$".format(j))
                    m = m + 1
            plt.legend()
        else:
            # We are plotting a two-fermion field
            if is_f_complex:
                for j in j_QD:
                    plt.plot(self.t, np.real(f[m,:]), label=field_labels[0]+"$_{{{:d},{:d}}}$".format(j[0],j[1]))
                    plt.plot(self.t, np.imag(f[m,:]), label=field_labels[1]+"$_{{{:d},{:d}}}$".format(j[0],j[1]))
                    plt.plot(self.t, np.abs(f[m,:]), label=field_labels[2]+"$_{{{:d},{:d}}}$".format(j[0],j[1]))
                    m = m + 1
            else:
                for j in j_QD:
                    plt.plot(self.t, f[m,:], label=field_labels[0]+"$_{{{:d},{:d}}}$".format(j[0],j[1]))
                    m = m + 1
            plt.legend()
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True) # Add a grid for better readability
        plt.tight_layout() # Adjust plot to ensure everything fits
        plt.show() # Display the plot       
        return
        
    def save_to_file(self, filename: str = "QD.pickle"):
        """
        Saves the current state of the QD object's p, t, and z attributes
        to a pickle file.

        Parameters
        ----------
        
            filename : string
            
              The name of the file to save the data to. Defaults to 
              "QD.pickle" if not provided.
        """
        data_to_save = {
            'p': self.p,
            'q': self.q,
            't': self.t,
            'z': self.z
        }
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Data saved successfully to {filename}")
        except Exception as e:
            print(f"Error saving data to pickle file: {e}")
            
    def set_detuning(self,Detuning_centre=0.0,std_Delta_nu=1,n_outliers=5,rn_seed=None):
        """
        This function sets the detuning for models with different
        quantum dots. These are identified by the fact that the string
        p['Model'] ends in "_d". The detuning are picked from a normal
        distribution with mean Delta_nu and standard deviation
        std_Delta_nu. Additionally, 2*n_outliers quantum dots are
        assigned detuning ±γ, ±2γ,…, ±n_outliers*γ. In case of
        distributions with small value of std_Delta_nu, these outliers
        allow the visualisation of the tails of the activity curve,
        i.e. of the graph of :math:`|\\langle v^{\dagger} c \\rangle|`
        as a function of detuning.
        
        The detuning in models with identical quantum dots can be set using
        the method set_parameter with argument 'Delta_nu'

        Parameters
        ----------
        
        Detuning_centre : real (default 0)
            Mean of the detuning distribution in units of γ.
            
        std_Delta_nu : real (strictly positive) (default 1)
           Standard deviation of the detuning distribution in units of γ.
           
        n_outliers : integer (positive, can be zero) (default 5)
           The number of outlier quantum dots, i.e. with pre-assigned detuning
           ±γ, ±2γ,…, ±n_outliers*γ.
           
        rn_seed : string or integer (default None)
            Seed of the random number generator for 
            the detuning of the quantum dots. Use None to have non-repeatable 
            distributions of quantum dot parameters. If rn_seed is a positive 
            integer then all simulations run with the same value of rn_seed will 
            have the same distribution of quantum dot parameters.

        Returns
        -------
        
        None. The detuning distribution is stored in the dictionary 
        p['Delta_nu']. Its parameters are stored in the dictionary p as
        p['Detuning_centre'] and p['std_Delta_nu'].
        
        Usage
        -----
        
        my_qd.set_detuning(0.0,1.0,5,None)
        
        Create a sets of p['N'] quantum dots with detuning distributed 
        according to a normal distribution with 0 mean and standard deviation
        equal to γ. There are ten outlier quantum dots with detuning equal to 
        ±γ, ±2γ,…, ±5γ. No seed has been specified and so the distribution is 
        not repeatable.

        my_qd.set_detuning(0.0,1.0,5,938)
        
        Create a sets of p['N'] quantum dots with detuning distributed 
        according to a normal distribution with 0 mean and standard deviation
        equal to γ. There are ten outlier quantum dots with detuning equal to 
        ±γ, ±2γ,…, ±5γ. The random number generator has seed 938. An identical
        distribution can be obtained by using the same seed in a subsequent call.
        
        """
        if self.p['Model'][-2:] != "_d":
            return None
        
        # Check that the number of outliers is reasonable
        n_outliers = int(np.floor(abs(n_outliers)))
        if 2*n_outliers> self.p['N']:
            print("The number of outliers, {0:d}, is too large. It should be smaller than {1:d}.".format(n_outliers,int(np.floor(self.p['N']/2))))
            return None
        
        # Store the parameters of the detuning distribution.
        self.p['Detuning_centre'] = Detuning_centre
        self.p['std_Delta_nu'] = abs(std_Delta_nu)
        self.p['rn_seed'] = rn_seed
            
        # Set the random generator 
        np.random.seed(rn_seed) 
        # Generate the detunings
        self.p['Delta_nu'] = self.p['gamma']*(Detuning_centre \
            + self.p['std_Delta_nu']*np.random.randn(self.p['N']))
        # Insert outliers 
        n_outliers = np.floor(abs(n_outliers))
        self.p['n_outliers'] = n_outliers
        outlier_values = self.p['gamma'] \
            *np.array([-np.arange(1,n_outliers+1),np.arange(1,n_outliers+1)]).flatten()
        self.p["Delta_nu"][0:outlier_values.size] = outlier_values
        # Sort the detuning (not needed for calculations, but makes life easier)
        self.p['Delta_nu'] = np.sort(self.p['Delta_nu'], axis=None)
        
    def set_init_cond(self, init_condition: typing.Union[str, np.ndarray]):
        """
        Sets the initial condition array z0 based on the input. If the input
        parameter is incorrect the initial condition is left unchanged.

        Parameters
        ----------
        
            init_condition : string or numpy array
              If a string, must be "random", in which case sets z0 to a 
              1D NumPy array of size 8 with all elements selected at 
              random within physical constraints. If a np.ndarray, z0 is set
              equal to this array.
        """
        
        if isinstance(init_condition, str):
            if init_condition == "random":
                z_ampl = 50.0
                # Ensure that each run has different initial conditions
                np.random.seed(int(datetime.now().timestamp())) # Use current timestamp for "shuffle"
                self.z0 = z_ampl*np.random.rand(self.p['nf'])
                # Ensure that the fermionic variables are bounded in [0,1];
                self.z0[self.p['fv']] = np.random.rand(len(self.p['fv']))/np.sqrt(2) 
                print(f"Initial condition z0 set to 'random': {self.z0}")
                # Specify that an initial condition has been set.
                self.init_cond_set = True
            else:
                print("Error: Invalid string input for init_condition. Must be 'random'.")
        elif isinstance(init_condition, np.ndarray):
            if init_condition.ndim == 1:
                self.z0 = init_condition
                print(f"Initial condition z0 set to provided NumPy array: {self.z0}")
                # Specify that an initial condition has been set.
                self.init_cond_set = True
            else:
                print(f"Error: Provided NumPy array for init_condition must be 1-dimensional with {self.p['nf']} elements..")
        else:
            print(f"Error: Invalid input type for init_condition. Must be a string ('random') or a 1D NumPy array with {self.p['nf']} elements.")

    def set_parameter(self, new_params: dict):
        """
        Sets or updates the values in the parameter dictionary 'p'.

        Parameters
        ----------
        
            new_params : dictionary
              A dictionary containing new key-value pairs to be added or 
              updated in 'p'. Only names of existing fields are accepted.
        """
        if not isinstance(new_params, dict):
            print("Error: Input for set_p must be a dictionary.")
            return

        updated_keys = []
        skipped_keys = []
        for key, value in new_params.items():
            # if key in self.p:
            if key in self.p['allowed_keys']:
                self.p[key] = value
                updated_keys.append(key)
            else:
                skipped_keys.append(key)
        
        if updated_keys:
            print(f"Parameters 'p' updated for keys: {', '.join(updated_keys)}.")
            # print(f" Current 'p': {self.p}")
            print("")
            # Erase any previous values of time and QD variables.
            self.t = np.array([])  # Initialize an empty 1D NumPy array for t
            self.z = np.array([[]])  # Initialize an empty 2D NumPy array for z
            self.z_eq = np.array([])  # Initialize an empty 1D NumPy array for the equilibrium solution
            print("Any previously stored values of the time and the QD fields have been erased.")
        if skipped_keys:
            print(f"Warning: The following keys were skipped because they either do not exist or cannot be changed using set_parameter: {', '.join(skipped_keys)}")
        if not updated_keys and not skipped_keys:
            print("No parameters provided to update.")
        
    def _SparsityJac(self,f, z, p, q):
        """
        Computes numerically the sparsity matrix of the Jacobian of the model
        coded by the function f.

        Parameters
        ----------
        
        f : function handle
          The model equations. This is the function passed to initial value
          solver to integrate the model and depends on t, z and p.

        z : numpy 1D array
          The amplitude of the fields

        p : dictionary
          The model parameters

        Returns
        -------
        
        Sparsity_J : sparse csc matrix
          The sparsity matrix of the Jacobian of f evaluated at t=0 and z,
          with parameters p.
        """

        # Indices of the non-zero elements of the jacobian in index notation
        knz = []
        # Compute the Jacobian numerically column by column
        nz = len(z)  # Use len(z) for NumPy arrays/lists
        dz = 1e-3
        t = 0
        # Elements with modulus larger than zero_threshold are considered non-zero
        zero_threshold = 1e-6
        # zero_threshold = np.finfo(float).eps

        # Ensure z is a mutable copy if it's passed as an immutable type
        z_mutable = np.array(z, dtype=complex if z.dtype == complex else float)

        for m in range(nz):  # Python uses 0-based indexing for loops
            # Compute the derivatives with respect to z(m), i.e. the m-th column of
            # the Jacobian
            old_z_m = z_mutable[m]

            if np.abs(old_z_m) < 1:
                z_mutable[m] = old_z_m + dz
                fp = f(t, z_mutable, p,q)
                z_mutable[m] = old_z_m - dz
                fm = f(t, z_mutable, p,q)
                J_col = (fp - fm) / (2 * dz)
            else:
                z_mutable[m] = old_z_m * (1 + dz)
                fp = f(t, z_mutable, p,q)
                z_mutable[m] = old_z_m * (1 - dz)
                fm = f(t, z_mutable, p,q)
                J_col = (fp - fm) / (2 * dz * old_z_m)
            
            # Restore the original value of z(m)
            z_mutable[m] = old_z_m

            # Identify the elements that are different from zero
            
            # Get the 0-based row indices where elements are non-zero in the current column
            non_zero_rows_in_col = np.where(np.abs(J_col) > zero_threshold)[0]
            
            # Append the linear indices (0-based) for the Jacobian matrix
            # For a column 'm', the linear indices are `row_idx + m * nz` in 0-based column-major.
            knz.extend(non_zero_rows_in_col*nz + m) 

        # Convert the linear indices to (row, col) subscripts
        # np.unravel_index requires the shape (rows, cols) and linear indices (0-based).
        # `unravel_index` with `order='C'` (row-major) will correctly
        # convert these linear indices back to (row, column) pairs.
        inz, jnz = np.unravel_index(knz, (nz, nz), order='C')

        # Create the sparse matrix
        # Sparse matrix takes (data, (row_indices, col_indices)).
        # We want a matrix of ones where the Jacobian is non-zero.
        data = np.ones(len(knz))
        Sparsity_J = sparse.csc_matrix((data, (inz, jnz)), shape=(nz, nz))

        return Sparsity_J

class QD_CIM(QD):
    """
    A Python object to store and manage data related to the Coherent-Incoherent
    Model with identical quantum dots (CIM). This class does not add any 
    methods to those of the parent class.

    See also: :py:class:`QD.QD`
    """

    def __init__(self):
        """
        Initializes the QD_CIM object with empty data structures.
        """
        super().__init__()
        # Model type
        self.p['Model'] = "CIM"
        # Set the number of equations
        self.p['nf'] = 8
        # Define a dictionary of indices that map the physical field to their
        # real and imaginary part format stored in the array z.
        self.q = {}
        self.q['j_bR'] = 0
        self.q['j_bI'] = 1
        self.q['j_Bb'] = 2
        self.q['j_Cc'] = 3
        self.q['j_VcR'] = 4
        self.q['j_VcI'] = 5
        self.q['j_bCvR'] = 6
        self.q['j_bCvI'] = 7
        # Set the indices of the fermionic variables
        self.p['fv'] = [self.q['j_Cc'],self.q['j_VcR'],self.q['j_VcI']]
        
    def eq(self,t, z, p, q):
        """
        Returns the time derivative of the Coherent-Incoherent Model
        with identical quantum dots (CIM), equations (S.11) of [3],
        written in real and imaginary part. The variable z passed to
        this function contains either the real and imaginary parts of
        the complex fields, or are the amplitude of the real
        fields. The mapping is as follows:

        | b = z[q['j_bR']] + 1j*z[q['j_bI']]
        | Bb = z[q['j_Bb']]
        | Cc = z[q['j_Cc']]
        | Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        | bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]

        where we have used the notation b= :math:`\\langle b \\rangle`
        and B= :math:`\\langle b^{\dagger}\\rangle`, etc.
        
        Parameters
        ----------
        
        t : real
          The time value at which the derivative is computed.
          
        z : 1D numpy array
          The current values of the model variables in real and imaginary form.
          
        p : dictionary
          The model's parameters
          
        q : dictionary
          The indices that map the physical fields to the z array

        Returns
        -------
        
        zdot : 1D numpy array
          The time derivatives at time t of the model variables z.

        """

        # The system variables in complex
        b = z[q['j_bR']] + 1j*z[q['j_bI']]
        Bb = z[q['j_Bb']]
        Cc = z[q['j_Cc']]
        Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]

        # The CIM equations for the expectation values [Mark's thesis, equations
        # (3.51) to (3.55), page 38, but written in a rotating frame]
        b_dot = -p['gamma_c']*b + p['N']*np.conj(p['g'])*Vc
        Vc_dot = -(p['gamma'] - 1j*p['Delta_nu'])*Vc + p['g']*b*(2*Cc - 1)
        Cc_dot = p['r']*(1 - Cc) - (p['gamma_nr'] + p['gamma_nl'])*Cc - 2*np.real(p['g']*bCv)
        Bb_dot = -2*p['gamma_c']*Bb + 2*p['N']*np.real(p['g']*bCv)
        bCv_dot = -(p['gamma'] + p['gamma_c'] + 1j*p['Delta_nu'])*bCv + \
                    np.conj(p['g'])*(Cc + Bb*(2*Cc - 1)) + \
                    np.conj(p['g'])*(p['N'] - 1)*np.conj(Vc)*Vc

        # Return them as a vector of real quantities in the SAME ORDER as 
        # the input z vector.
        z_dot = np.array([
            np.real(b_dot),
            np.imag(b_dot),
            Bb_dot,
            Cc_dot,
            np.real(Vc_dot),
            np.imag(Vc_dot),
            np.real(bCv_dot),
            np.imag(bCv_dot)
        ])

        return z_dot
    
class QD_CIM_d(QD):
    """
    A Python object to store and manage data related to Coherent-Incoherent
    Model of quantum dots with different quantum dots (CIM_d). This class 
    does not add any public methods to those of the parent class, but overrides
    the set_parameter method to ensure that if the number of quantum dots is 
    changed then all the parameters and indices that depend on this value are
    updated.

    See also: :py:class:`QD.QD`
    
    """

    def __init__(self):
        """
        Initializes the QD_CIM object with empty data structures.
        """
        super().__init__()
        # Model type
        self.p['Model'] = "CIM_d"
        # Remove 'Delta_nu' from the list of parameters that the user can change
        self.p['allowed_keys'] = [s for s in self.p['allowed_keys'] if s != 'Delta_nu']
        # Set the parameters that depend on the number of quantum dots
        self.__set_N_dep_params__()

    def __set_N_dep_params__(self):
        """
        Function to set the parameters and indices that depend on the number of
        quantum dots.

        Returns
        -------
        None.

        """
        # Set the number of equations
        self.p['nf'] = 5*self.p['N'] + 3
        # Define a dictionary of indices that map the physical field to their
        # real and imaginary part format stored in the array z.
        self.q = {}
        # Indices of the various fields in the vector of variables.
        self.q['j_bR'] = 0
        self.q['j_bI'] = 1
        self.q['j_Bb'] = 2
        self.q['j_Cc'] = 3 + np.arange(self.p['N'])
        self.q['j_VcR'] = 3 + self.p['N'] + np.arange(self.p['N'])
        self.q['j_VcI'] = 3 + 2*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCvR'] = 3 + 3*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCvI'] = 3 + 4*self.p['N'] + np.arange(self.p['N'])
        # Set the indices of the fermionic variables
        self.p['fv'] = np.array([self.q['j_Cc'],self.q['j_VcR'],self.q['j_VcI']]).flatten()
        # The gain coefficient is now a vector (with all identical values)
        self.p['g_d'] = self.p['g']*np.ones(self.p['N'])
        # Set the detuning distribution 
        if 'Detuning_centre' in self.p.keys():
            # Use the old parameters
            self.set_detuning(self.p['Detuning_centre'],self.p['std_Delta_nu'],self.p['n_outliers'],self.p['rn_seed'])
        else:  
            #use default parameters
            self.set_detuning()
            
    def eq(self,t, z, p, q):
        """
        Returns the time derivative of the Coherent-Incoherent Model
        with different quantum dots (CIM_d), equations (S.10) of [3],
        written in real and imaginary part. The variable z passed to
        this function contains either the real and imaginary parts of
        the complex fields, or are the amplitude of the real
        fields. The mapping is as follows: 

        | b = z[q['j_bR']] + 1j*z[q['j_bI']]
        | Bb = z[q['j_Bb']]
        | Cc = z[q['j_Cc']]
        | Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        | bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]

        where we have used the notation b= :math:`\\langle b \\rangle`
        and B= :math:`\\langle b^{\dagger}\\rangle`, etc.
        
        Parameters
        ----------
        
        t : real
          The time value at which the derivative is computed.
          
        z : 1D numpy array
          The current values of the model variables in real and imaginary form.
          
        p : dictionary
          The model's parameters
          
        q : dictionary
          The indices that map the physical fields to the z array

        Returns
        -------
        
        zdot : 1D numpy array
          The time derivatives at time t of the model variables z.
          
        """

        # The system variables in complex
        b = z[q['j_bR']] + 1j*z[q['j_bI']]
        Bb = z[q['j_Bb']]
        Cc = z[q['j_Cc']]
        Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]

        # The CIM equations for the expectation values

        b_dot = -p['gamma_c']*b + p['g_d'].conj().dot(Vc)

        Bb_dot = -2*p['gamma_c']*Bb + 2*np.real(p['g_d'].dot(bCv))

        Vc_dot = -(p['gamma']-1j*p['Delta_nu'])*Vc + p['g_d']*b*(2*Cc-1)

        Cc_dot = p['r']*(1-Cc) - (p['gamma_nl']+p['gamma_nr'])*Cc \
             - 2*np.real(p['g_d']*bCv)

        bCv_dot = -(p['gamma']+p['gamma_c']+1j*p['Delta_nu'])*bCv \
              + p['g_d'].conj()*(Cc + Bb*(2*Cc-1)) \
              + np.conj(Vc)*((p['g_d'].conj()).dot(Vc) - p['g_d'].conj()*Vc)

        # Package all the time derivatives in a single vector
        z_dot = np.concatenate((
            [np.real(b_dot), np.imag(b_dot), Bb_dot],
            Cc_dot,
            np.real(Vc_dot), np.imag(Vc_dot),
            np.real(bCv_dot), np.imag(bCv_dot)
            ))

        return z_dot
    
    def set_parameter(self, new_params: dict):
        """
        Sets or updates the values in the parameter dictionary 'p'. It checks
        if the number of quantum dots has changed and regenerates all the 
        parameters and indices that depend on this value.

        Parameters
        ----------
        
            new_params : dictionary
              A dictionary containing new key-value pairs to be added or 
              updated in 'p'. Only names of existing fields are accepted.
        """
        
        # Store the previous value of the number of quantum dots
        N_old = self.p['N']
        # Store the previous value of the gain coefficient
        g_old = self.p['g']
        
        # Update the parameters
        super().set_parameter(new_params)
        # Check if N has changed and update the N-dependent parameters if needed
        if self.p['N'] != N_old:
            self.__set_N_dep_params__()
        # Check if the gain coefficient has changed
        if self.p['g'] != g_old:
            # Generate again the gain coefficients
            self.p['g_d'] = self.p['g']*np.ones(self.p['N'])
        
class QD_TPM(QD):
    """
    A Python object to store and manage data related to Two-Particle Correlation
    Model (TPM) of quantum dots. This class does not add any methods to those
    of the parent class.

    See also: :py:class:`QD.QD`
    
    """

    def __init__(self):
        """
        Initializes the QD_CIM object with empty data structures.
        """
        super().__init__()
        # Model type
        self.p['Model'] = "TPM"
        # Set the number of equations
        self.p['nf'] = 21
        # Define a dictionary of indices that map the physical field to their
        # real and imaginary part format stored in the array z.
        self.q = {}
        self.q['j_bR'] = 0
        self.q['j_bI'] = 1
        self.q['j_Bb'] = 2
        self.q['j_Cc'] = 3
        self.q['j_VcR'] = 4
        self.q['j_VcI'] = 5
        self.q['j_bCvR'] = 6
        self.q['j_bCvI'] = 7
        self.q['j_bCcR'] = 8
        self.q['j_bCcI'] = 9
        self.q['j_bVcR'] = 10
        self.q['j_bVcI'] = 11
        self.q['j_bbR'] = 12
        self.q['j_bbI'] = 13
        self.q['j_CVcvR'] = 14
        self.q['j_CVcvI'] = 15
        self.q['j_VCccR'] = 16
        self.q['j_VCccI'] = 17
        self.q['j_VVccR'] = 18
        self.q['j_VVccI'] = 19
        self.q['j_CCcc'] = 20
        # Set the indices of the fermionic variables
        self.p['fv'] = [self.q['j_Cc'],self.q['j_VcR'],self.q['j_VcI'],\
                        self.q['j_CVcvR'], self.q['j_CVcvI'], \
                        self.q['j_VCccR'], self.q['j_VCccI'], \
                        self.q['j_VVccR'], self.q['j_VVccI'], \
                        self.q['j_CCcc']]
        
    def eq(self,t, z, p, q):
        """
        Returns the time derivative of the Two Particle Model with
        identical quantum dots and correlation damping coefficient μγ
        with identical quantum dots (TPM), equations (S.7) of [3],
        written in real and imaginary part. The equations are written
        in a frame rotating with the coherent field. The detuning is
        Δν=ν-Δε, with ν the field cavity frequency and Δε the quantum
        dot frequency (difference between energy levels).  The
        variable z passed to this function contains either the real
        and imaginary parts of the complex fields, or are the
        amplitude of the real fields. The mapping is as follows:
       
        | b = z[q['j_bR']] + 1j*z[q['j_bI']]
        | Bb = z[q['j_Bb']]
        | Cc = z[q['j_Cc']]
        | Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        | bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]
        | bCc = z[q['j_bCcR']] + 1j*z[q['j_bCcI']]
        | bVc = z[q['j_bVcR']] + 1j*z[q['j_bVcI']]
        | bb = z[q['j_bbR']] + 1j*z[q['j_bbI']]
        | CVcv = z[q['j_CVcvR']]+1j*z[q['j_CVcvI']] 
        | VCcc = z[q['j_VCccR']]+1j*z[q['j_VCccI']]
        | VVcc = z[q['j_VVccR']]+1j*z[q['j_VVccI']]
        | CCcc = z[q['j_CCcc']]

        where we have used the notation b= :math:`\\langle b \\rangle`
        and B= :math:`\\langle b^{\dagger}\\rangle`, etc.        
        
        Parameters
        ----------
        
        t : float
            Time (unused in this function, but required by ODE solvers).
            
        z : numpy.ndarray
            A 1D numpy array containing the real and imaginary parts
            of the complex fields, or amplitudes of the real fields,
            stacked.
            
        p : dict
            A dictionary containing the equation coefficients.
            
        q : dictionary
            The indices that map the physical fields to the z array

        Returns
        -------
        
        z_dot : numpy.ndarray
            A 1D numpy array containing the time derivatives of the corresponding
            elements in z, stacked as real and imaginary parts.
        """

        # Variable mapping from the input array z
        b = z[q['j_bR']] + 1j*z[q['j_bI']]
        Bb = z[q['j_Bb']]
        Cc = z[q['j_Cc']]
        Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]
        bCc = z[q['j_bCcR']]+1j*z[q['j_bCcI']]
        bVc = z[q['j_bVcR']]+1j*z[q['j_bVcI']]
        bb = z[q['j_bbR']]+1j*z[q['j_bbI']]
        CVcv = z[q['j_CVcvR']]+1j*z[q['j_CVcvI']] 
        VCcc = z[q['j_VCccR']]+1j*z[q['j_VCccI']]
        VVcc = z[q['j_VVccR']]+1j*z[q['j_VVccI']]
        CCcc = z[q['j_CCcc']]

        # The TPM equations for the expectation values

        # Equation (49)
        Cc_dot = p['r']*(1-Cc) - (p['gamma_nr']+p['gamma_nl'])*Cc - 2*np.real(p['g']*bCv)
        
        # Equation (50)
        Bb_dot = -2*p['gamma_c']*Bb + 2*p['N']*np.real(p['g']*bCv)
        
        # Equation (51)
        bCv_dot = -(p['gamma']+p['gamma_c']+1j*p['Delta_nu'])*bCv \
                  + p['g']*(Cc + Bb*(2*Cc-1) + 4*np.real(b*np.conj(bCc)) - 4*b*np.conj(b)*Cc) \
                  + p['g']*(p['N']-1)*CVcv
        
        # Equation (52)
        b_dot = -p['gamma_c']*b + p['N']*p['g']*Vc
        
        # Equation (53)
        Vc_dot = -(p['gamma']-1j*p['Delta_nu'])*Vc + p['g']*(2*bCc - b)
        
        # Equation (54)
        bCc_dot = -(p['gamma_nr']+p['gamma_c'])*bCc - p['gamma_nl']*b*Cc \
                  - p['g']*(2*b*(bCv - b*np.conj(Vc)) + bb*np.conj(Vc)) \
                  - p['g']*(np.conj(b)*(bVc - 2*b*Vc) + Bb*Vc + b*np.conj(bCv)) \
                  + p['r']*b*(1-Cc) + p['g']*(p['N']-1)*VCcc
        
        # Equation (55)
        bVc_dot = -(p['gamma']+p['gamma_c']-1j*p['Delta_nu'])*bVc \
                  + p['g']*(bb*(2*Cc-1) + 4*b*(bCc - b*Cc)) \
                  + (p['N']-1)*p['g']*VVcc
        
        # Equation (56)
        bb_dot = -2*p['gamma_c']*bb + 2*p['N']*np.conj(p['g'])*bVc
        
        # Equation (57)
        CVcv_dot = -2*(1+p['mu'])*p['gamma']*CVcv \
                   - p['g']*(-2*np.conj(b)*VCcc + np.conj(bCv)*(1-2*Cc) \
                             - 2*Vc*np.conj(bCc) + 4*np.conj(b)*Cc*Vc) \
                   - p['g']*(bCv*(1-2*Cc) - 2*b*np.conj(VCcc) - 2*np.conj(Vc)*bCc \
                             + 4*b*Cc*np.conj(Vc))
        
        # Equation (58)
        VCcc_dot = -((1+p['mu'])*p['gamma']+p['gamma_nr']-1j*p['Delta_nu'])*VCcc \
                   - p['g']*(-2*b*CCcc + 4*b*Cc*Cc - 2*Cc*bCc - 2*Cc*bCc + bCc) \
                   + p['g']*(-b*CVcv + 2*b*Vc*np.conj(Vc) - Vc*bCv - np.conj(Vc)*bVc) \
                  + np.conj(p['g'])*(-np.conj(b)*VVcc + 2*np.conj(b)*Vc*Vc \
                                     - Vc*np.conj(bCv) - Vc*np.conj(bCv)) \
                   + p['r']*Vc*(1-Cc) - p['gamma_nl']*Vc*Cc
        
        # Equation (59)
        CCcc_dot = -2*p['gamma_nr']*CCcc \
                   + 2*np.real(p['g']*(-b*np.conj(VCcc) + 2*b*np.conj(Vc)*Cc \
                                       - Cc*bCv - np.conj(Vc)*bCc) \
                               + p['g']*(-b*np.conj(VCcc) + 2*b*Cc*np.conj(Vc) \
                                         - Cc*bCv - np.conj(Vc)*bCc)) \
                  + p['r']*(Cc*(1-Cc) + Cc*(1-Cc)) - 2*p['gamma_nl']*Cc*Cc
        
        # Equation (60)
        VVcc_dot = -2*((1+p['mu'])*p['gamma']-1j*p['Delta_nu'])*VVcc \
                   - p['g']*(-2*b*VCcc + 4*b*Cc*Vc - 2*Cc*bVc - 2*Vc*bCc + bVc) \
                  - p['g']*(-2*b*VCcc  - 2*Vc*bCc - 2*Cc*bVc + 4*b*Cc*Vc + bVc)

        # Return them as a vector of real quantities in the SAME ORDER as 
        # the input z vector.
        z_dot = np.array([
            np.real(b_dot), np.imag(b_dot),
            Bb_dot,
            Cc_dot,
            np.real(Vc_dot), np.imag(Vc_dot),
            np.real(bCv_dot), np.imag(bCv_dot),
            np.real(bCc_dot), np.imag(bCc_dot),
            np.real(bVc_dot), np.imag(bVc_dot),
            np.real(bb_dot), np.imag(bb_dot),
            np.real(CVcv_dot), np.imag(CVcv_dot),
            np.real(VCcc_dot), np.imag(VCcc_dot),
            np.real(VVcc_dot), np.imag(VVcc_dot),
            CCcc_dot
        ])

        return z_dot


class QD_TPM_d(QD):
    """
    A Python object to store and manage data related to the
    Two-Particle Correlation Model of quantum dots with different
    quantum dots (TPM_d). This class does not add any public methods
    to those of the parent class, but overrides the set_parameter
    method to ensure that if the number of quantum dots is changed
    then all the parameters and indices that depend on this value are
    updated.

    See also: :py:class:`QD.QD`
    
    """

    def __init__(self):
        """
        Initializes the QD_TPM_d object with empty data structures.
        """
        super().__init__()
        # Model type
        self.p['Model'] = "TPM_d"
        # Remove 'Delta_nu' from the list of parameters that the user can change
        self.p['allowed_keys'] = [s for s in self.p['allowed_keys'] if s != 'Delta_nu']
        # Set the parameters that depend on the number of quantum dots
        self.__set_N_dep_params__()
        
    def __set_N_dep_params__(self):
        """
        Function to set the parameters and indices that depend on the number of
        quantum dots.

        Returns
        -------
        None.

        """
        
        # Set the number of equations
        self.p['nf'] = 5 + 9*self.p['N'] + 7*self.p['N']*(self.p['N'] - 1)
        # Define a dictionary of indices that map the physical field to their
        # real and imaginary part format stored in the array z.
        self.q = {}
        # Indices of the various fields in the vector of variables.
        self.q['j_bR'] = 0
        self.q['j_bI'] = 1
        self.q['j_Bb'] = 2
        self.q['j_bbR'] = 3
        self.q['j_bbI'] = 4
        self.q['j_Cc'] = 5 + np.arange(self.p['N'])
        self.q['j_VcR'] = 5 + self.p['N'] + np.arange(self.p['N'])
        self.q['j_VcI'] = 5 + 2*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCvR'] = 5 + 3*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCvI'] = 5 + 4*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCcR'] = 5 + 5*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCcI'] = 5 + 6*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bVcR'] = 5 + 7*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bVcI'] = 5 + 8*self.p['N'] + np.arange(self.p['N'])
        
        # Helper function for the indices of the two-fermion operators
        def CreateIndexArray(m):
           jj = np.zeros((self.p['N'], self.p['N']), dtype=int)
           offset = 5 + 9*self.p['N'] + m*self.p['N']*(self.p['N']-1)
           current_idx = 0
           for k in range(self.p['N']):
             # Elements before the diagonal in the current row
             for r_idx in range(k):
               jj[k,r_idx] = offset + current_idx
               current_idx += 1
             # Elements after the diagonal in the current row
             for r_idx in range(k + 1, self.p['N']):
               jj[k,r_idx] = offset + current_idx
               current_idx += 1
           return jj
        
        # Indices on the fermionic variables in matrix form (for Jacobian)
        self.q['jm_CVcvR'] = CreateIndexArray(0)
        self.q['jm_CVcvI'] = CreateIndexArray(1)
        self.q['jm_VCccR'] = CreateIndexArray(2)
        self.q['jm_VCccI'] = CreateIndexArray(3)
        self.q['jm_VVccR'] = CreateIndexArray(4)
        self.q['jm_VVccI'] = CreateIndexArray(5)
        self.q['jm_CCcc'] = CreateIndexArray(6)
       
        # Indices on the fermionic variables in vector form (for equation)
        self.q['j_CVcvR'] = self.q['jm_CVcvR'][self.q['jm_CVcvR'] != 0]
        self.q['j_CVcvI'] = self.q['jm_CVcvI'][self.q['jm_CVcvI'] != 0]
        self.q['j_VCccR'] = self.q['jm_VCccR'][self.q['jm_VCccR'] != 0]
        self.q['j_VCccI'] = self.q['jm_VCccI'][self.q['jm_VCccI'] != 0]
        self.q['j_VVccR'] = self.q['jm_VVccR'][self.q['jm_VVccR'] != 0]
        self.q['j_VVccI'] = self.q['jm_VVccI'][self.q['jm_VVccI'] != 0]
        self.q['j_CCcc'] = self.q['jm_CCcc'][self.q['jm_CCcc'] != 0]
       
        # Indices of the non-zero elements of a two-fermion array
        rows_nz, cols_nz = np.where(self.q['jm_CVcvR'] != 0) # Assumes all jm_ matrices have same sparsity pattern
        self.q['jnz'] = np.ravel_multi_index((rows_nz, cols_nz), (self.p['N'], self.p['N'])) # 0-indexed linear indices
       
        # Indices of the diagonal elements of the two fermion operators
        rows = np.arange(self.p['N'])
        self.q['jd'] = np.ravel_multi_index((rows, rows), (self.p['N'], self.p['N'])) 

        # Set the indices of the fermionic variables
        self.p['fv'] = np.concatenate([self.q['j_Cc'],self.q['j_VcR'],self.q['j_VcI'],\
                        self.q['j_CVcvR'], self.q['j_CVcvI'], \
                        self.q['j_VCccR'], self.q['j_VCccI'], \
                        self.q['j_VVccR'], self.q['j_VVccI'], \
                        self.q['j_CCcc']])
        # The gain coefficient is now a vector (with all identical values)
        self.p['g_d'] = self.p['g']*np.ones(self.p['N'])
        # Set the detuning distribution 
        if 'Detuning_centre' in self.p.keys():
            # Use the old parameters
            self.set_detuning(self.p['Detuning_centre'],self.p['std_Delta_nu'],self.p['n_outliers'],self.p['rn_seed'])
        else:  
            #use default parameters
            self.set_detuning()
        
    def eq(self,t, z, p, q):
        """
        Returns the time derivative of the Two Particle Model with
        non-identical quantum dots and correlation damping coefficient
        μγ (TPM_d), equations (S.5) of [3], written in real and
        imaginary part.  The equations are written in a frame rotating
        with the coherent field. The detuning is Δν=ν-Δε, with ν the
        field cavity frequency and Δε the quantum dot frequency
        (difference between energy levels).  The variable z passed to
        this function contains either the real and imaginary parts of
        the complex fields, or are the amplitude of the real
        fields. The mapping is as follows:
       
        | b = z[q['j_bR']] + 1j*z[q['j_bI']]
        | Bb = z[q['j_Bb']]
        | Cc = z[q['j_Cc']]
        | Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        | bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]
        | bCc = z[q['j_bCcR']] + 1j*z[q['j_bCcI']]
        | bVc = z[q['j_bVcR']] + 1j*z[q['j_bVcI']]
        | bb = z[q['j_bbR']] + 1j*z[q['j_bbI']]

        where we have used the notation b= :math:`\\langle b \\rangle`
        and B= :math:`\\langle b^{\dagger}\\rangle`, etc.  See the
        body of the function for the mapping of the two-fermion
        operators. For the purpose of this function they are
        represented as matrices whose values are stored in the 1D
        numpy array z.
    
        Parameters
        ----------
        
        t : float
            Time (unused in this function, but required by ODE solvers).
            
        z : numpy.ndarray
            A 1D numpy array containing the real and imaginary parts of the complex
            fields, or amplitudes of the real fields, stacked.
            
        p : dict
            A dictionary containing the equation coefficients.
            
        q : dictionary
            The indices that map the physical fields to the z array

        Returns
        -------
        
        z_dot : numpy.ndarray
            A 1D numpy array containing the time derivatives of the corresponding
            elements in z, stacked as real and imaginary parts.

        """
        
        # Number of quantum dots
        n = p['N']

        # CIM variables
        b = z[q['j_bR']] + 1j*z[q['j_bI']]
        Bb = z[q['j_Bb']]
        Cc = z[q['j_Cc']]
        Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]

        # Boson-fermion and boson-boson operators
        bCc = z[q['j_bCcR']] + 1j*z[q['j_bCcI']]
        bVc = z[q['j_bVcR']] + 1j*z[q['j_bVcI']]
        bb = z[q['j_bbR']] + 1j*z[q['j_bbI']]

        # Two-Particle Fermionic operators.
        CVcv = np.zeros(n*n, dtype=complex)
        CVcv[q['jnz']] = z[q['j_CVcvR']] + 1j*z[q['j_CVcvI']]
        CVcv = CVcv.reshape((n, n))

        VCcc = np.zeros(n*n, dtype=complex)
        VCcc[q['jnz']] = z[q['j_VCccR']] + 1j*z[q['j_VCccI']]
        VCcc = VCcc.reshape((n, n))

        VVcc = np.zeros(n*n, dtype=complex)
        VVcc[q['jnz']] = z[q['j_VVccR']] + 1j*z[q['j_VVccI']]
        VVcc = VVcc.reshape((n, n))

        CCcc = np.zeros(n*n, dtype=float) # CCcc is real
        CCcc[q['jnz']] = z[q['j_CCcc']]
        CCcc = CCcc.reshape((n, n))

        # The TPM equations for the expectation values

        b_dot = -p['gamma_c']*b + p['g_d'].conj().dot(Vc)

        Bb_dot = -2*p['gamma_c']*Bb + 2*np.real(p['g_d'].dot(bCv))

        Vc_dot = -(p['gamma'] - 1j*p['Delta_nu'])*Vc + p['g_d']*(2*bCc - b)

        Cc_dot = p['r']*(1 - Cc) - (p['gamma_nl'] + p['gamma_nr'])*Cc - 2*np.real(p['g_d']*bCv)

        bCv_dot = -(p['gamma'] + p['gamma_c'] + 1j*p['Delta_nu'])*bCv \
                  + p['g_d'].conj()*(Cc + Bb*(2*Cc - 1) - 4*b*np.conj(b)*Cc + 4*np.real(b*np.conj(bCc))) \
                  + (CVcv.T).dot(p['g_d'].conj())

        bCc_dot = -(p['gamma_c'] + p['gamma_nr'])*bCc - p['gamma_nl']*b*Cc \
                  - p['g_d']*(np.conj(Vc)*bb + 2*b*bCv - 2*(b**2)*np.conj(Vc)) \
                  - p['g_d'].conj()*(Vc*Bb + b*np.conj(bCv) - 2*(np.abs(b)**2)*Vc \
                  + np.conj(b)*bVc) + VCcc.dot(p['g_d'].conj()) + p['r']*b*(1 - Cc)

        bVc_dot = -(p['gamma_c'] + p['gamma'] - 1j*p['Delta_nu'])*bVc \
                  + p['g_d']*(bb*(2*Cc - 1) + 4*b*bCc - 4*Cc*b**2) \
                  + VVcc.dot(p['g_d'].conj())

        bb_dot = -2*p['gamma_c']*bb + 2*p['g_d'].dot(bVc)

        # Two-fermion operators

        # 1) This is an incoherent variable and as such can be nonzero below threshold
        CVcv_dot = -(2*p['gamma']*(1 + p['mu']) - 1j*p['Delta_minus'])*CVcv \
                   - p['g_d'].conj()*(-2*np.conj(b)*VCcc.T + np.outer(np.conj(bCv), (1-2*Cc)) \
                   + 2*np.outer(Vc, (-bCc + 2*np.conj(b)*Cc))) \
                   - np.multiply(p['g_d'][:,np.newaxis],-2*b*np.conj(VCcc) + np.outer((1-2*Cc),bCv) \
                   + 2*np.outer((-bCc+2*b*Cc),Vc))

        # 2) EV of the product of standard polarisation and carrier density - coherent
        VCcc_dot = -(p['gamma']*(1 + p['mu']) + p['gamma_nr'] - 1j*p['Delta_nu'].T)*VCcc \
                   - p['g_d']*(-2*b*CCcc + 4*np.outer(b*Cc,Cc) + np.outer(bCc,(1-2*Cc)) - 2*np.outer(Cc, bCc)) \
                   + np.multiply(p['g_d'][:,np.newaxis],-b*(CVcv.T) + 2*np.outer(b*np.conj(Vc), Vc) - np.outer(bCv, Vc) - np.outer(np.conj(Vc), bVc)) \
                   + np.multiply(p['g_d'].conj()[:,np.newaxis],-np.conj(b)*VVcc + 2*np.outer(np.conj(b)*Vc,Vc) - np.outer(np.conj(bCv), Vc) - np.outer(Vc, bCv)) \
                   + np.outer(p['r']*(1-Cc), Vc) - np.outer(p['gamma_nl']*Cc, Vc)

        # 3) EV of the product between standard polarisation between different QDs
        VVcc_dot = -2*(p['gamma']*(1 + p['mu']) - 1j*p['Delta_plus'])*VVcc \
                   - p['g_d']*(-2*b*(VCcc.T) + 4*np.outer(b*Vc,Cc) + np.outer(bVc,(1-2*Cc)) - 2*np.outer(Vc,bCc)) \
                   - np.multiply(p['g_d'][:,np.newaxis],-2*b*VCcc - 2*np.outer(bCc,Vc) + np.outer((1-2*Cc), bVc) + 4*np.outer(b*Cc,Vc))

        # 4) EV of the product between carrier density between different QDs
        CCcc_dot = -2*p['gamma_nr']*CCcc \
                   + 2*np.real( \
                       p['g_d']*(-b*np.conj(VCcc) + 2*np.outer(b*Cc,Vc.conj()) - np.outer(Cc,bCv) - np.outer(bCc,Vc.conj())) \
                       + np.multiply(p['g_d'][:,np.newaxis],-b*VCcc.conj().T + 2*np.outer(b*Vc.conj(),Cc) - np.outer(bCv,Cc) - np.outer(Vc.conj(), bCc)) \
                   ) \
                   + p['r']*(np.outer(Cc,(1-Cc)) + np.outer((1-Cc),Cc)) - 2*np.outer(p['gamma_nl']*Cc,Cc)

        # Package all the time derivatives in a single vector
        z_dot = np.concatenate((
            [np.real(b_dot), np.imag(b_dot), Bb_dot, np.real(bb_dot), np.imag(bb_dot)],
            Cc_dot,
            np.real(Vc_dot), np.imag(Vc_dot),
            np.real(bCv_dot), np.imag(bCv_dot),
            np.real(bCc_dot), np.imag(bCc_dot),
            np.real(bVc_dot), np.imag(bVc_dot),
            np.real(CVcv_dot.flatten()[q['jnz']]), np.imag(CVcv_dot.flatten()[q['jnz']]),
            np.real(VCcc_dot.flatten()[q['jnz']]), np.imag(VCcc_dot.flatten()[q['jnz']]),
            np.real(VVcc_dot.flatten()[q['jnz']]), np.imag(VVcc_dot.flatten()[q['jnz']]),
            CCcc_dot.flatten()[q['jnz']]
        ))

        return z_dot
    
    def set_detuning(self,Detuning_centre=0.0,std_Delta_nu=1,n_outliers=5,rn_seed=None):
        """        
        Modification of the QD.set_detuning method that sets also a
        couple of detuning related elements of the parameter
        dictionary p that are specific to the TPM_d equations.
        
        See also: :py:meth:`QD.set_detuning`
        
        """
        super().set_detuning(Detuning_centre,std_Delta_nu,n_outliers,rn_seed)
        self.p['Delta_minus'] = self.p['Delta_nu'][:, np.newaxis] - self.p['Delta_nu'][np.newaxis, :]
        self.p['Delta_plus'] = self.p['Delta_nu'][:, np.newaxis] + self.p['Delta_nu'][np.newaxis, :]

    def set_parameter(self, new_params: dict):
        """
        Sets or updates the values in the parameter dictionary 'p'. It checks
        if the number of quantum dots has changed and regenerates all the 
        parameters and indices that depend on this value.

        Parameters
        ----------
        
            new_params : dictionary
              A dictionary containing new key-value pairs to be added or 
              updated in 'p'. Only names of existing fields are accepted.
        """
        
        # Store the previous value of the number of quantum dots
        N_old = self.p['N']
        # Store the previous value of the gain coefficient
        g_old = self.p['g']
        
        # Update the parameters
        super().set_parameter(new_params)
        # Check if N has changed and update the N-dependent parameters if needed
        if self.p['N'] != N_old:
            self.__set_N_dep_params__()
        # Check if the gain coefficient has changed
        if self.p['g'] != g_old:
            # Generate again the gain coefficients
            self.p['g_d'] = self.p['g']*np.ones(self.p['N'])
            
class QD_TPM_1F(QD):
    """
    A Python object to store and manage data related to Two-Particle Correlation
    Model with no two particle fermionic correlations (TPM_1F) of quantum dots. 
    This class does not add any methods to those of the parent class.
   
    See also: :py:class:`QD.QD`
 
    """

    def __init__(self):
        """
        Initializes the QD_CIM object with empty data structures.
        """
        super().__init__()
        # Model type
        self.p['Model'] = "TPM_1F"
        # Set the number of equations
        self.p['nf'] = 14
        # Define a dictionary of indices that map the physical field to their
        # real and imaginary part format stored in the array z.
        self.q = {}
        self.q['j_bR'] = 0
        self.q['j_bI'] = 1
        self.q['j_Bb'] = 2
        self.q['j_Cc'] = 3
        self.q['j_VcR'] = 4
        self.q['j_VcI'] = 5
        self.q['j_bCvR'] = 6
        self.q['j_bCvI'] = 7
        self.q['j_bCcR'] = 8
        self.q['j_bCcI'] = 9
        self.q['j_bVcR'] = 10
        self.q['j_bVcI'] = 11
        self.q['j_bbR'] = 12
        self.q['j_bbI'] = 13
        # Set the indices of the fermionic variables
        self.p['fv'] = [self.q['j_Cc'],self.q['j_VcR'],self.q['j_VcI']]
        
    def eq(self,t, z, p, q):
        """
        Returns the time derivative of the Two Particle Model with the
        one particle approximation for Fermions and identical quantum
        dots (TPM_1F), equations (S.9) of [3], written in real and
        imaginary part.  The variable z passed to this function
        contains either the real and imaginary parts of the complex
        fields, or are the amplitude of the real fields. The mapping
        is as follows:
       
        | b = z[q['j_bR']] + 1j*z[q['j_bI']]
        | Bb = z[q['j_Bb']]
        | Cc = z[q['j_Cc']]
        | Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        | bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]
        | bCc = z[q['j_bCcR']] + 1j*z[q['j_bCcI']]
        | bVc = z[q['j_bVcR']] + 1j*z[q['j_bVcI']]
        | bb = z[q['j_bbR']] + 1j*z[q['j_bbI']]

        Parameters
        ----------
        
        t : float
            Time (unused in this function, but required by ODE solvers).
            
        z : numpy.ndarray
            A 1D numpy array containing the real and imaginary parts of the complex
            fields, or amplitudes of the real fields, stacked.
            
        p : dict
            A dictionary containing the equation coefficients.
            
        q : dictionary
            The indices that map the physical fields to the z array

        Returns
        -------
        
        z_dot : numpy.ndarray
            A 1D numpy array containing the time derivatives of the corresponding
            elements in z, stacked as real and imaginary parts.

        """

        # Variable mapping from the input array z
        b = z[q['j_bR']] + 1j*z[q['j_bI']]
        Bb = z[q['j_Bb']]
        Cc = z[q['j_Cc']]
        Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]
        bCc = z[q['j_bCcR']]+1j*z[q['j_bCcI']]
        bVc = z[q['j_bVcR']]+1j*z[q['j_bVcI']]
        bb = z[q['j_bbR']]+1j*z[q['j_bbI']]

        # The TPM_1F equations for the expectation values
        b_dot = -p['gamma_c']*b + p['N']*np.conj(p['g'])*Vc
        
        Vc_dot = -(p['gamma']-1j*p['Delta_nu'])*Vc + p['g']*(2*bCc-b)
        
        Cc_dot = p['r']*(1-Cc) - (p['gamma_nr']+p['gamma_nl'])*Cc \
                 - 2*np.real(p['g']*bCv)
                 
        Bb_dot = -2*p['gamma_c']*Bb + 2*p['N']*np.real(p['g']*bCv)
                 
        bCv_dot = -(p['gamma']+p['gamma_c']+1j*p['Delta_nu'])*bCv \
                  + np.conj(p['g'])*(Cc + Bb*(2*Cc-1) + 4*np.real(b*np.conj(bCc)) \
                                     - 4*b*np.conj(b)*Cc) \
                  + np.conj(p['g'])*(p['N']-1)*Vc*np.conj(Vc)

        bCc_dot = -(p['gamma_nr']+p['gamma_c'])*bCc - b*Cc*p['gamma_nl'] \
                  - p['g']*(np.conj(Vc)*bb + 2*b*bCv - 2*b*b*np.conj(Vc)) \
                  - np.conj(p['g'])*(Vc*Bb + b*np.conj(bCv) - 2*np.conj(b)*b*Vc \
                                     + np.conj(b)*bVc) \
                  + p['r']*b*(1-Cc) + np.conj(p['g'])*(p['N']-1)*Vc*Cc

        bVc_dot = -(p['gamma']+p['gamma_c']-1j*p['Delta_nu'])*bVc \
                  + p['g']*(bb*(2*Cc-1) + 4*b*(bCc-b*Cc)) \
                  + (p['N']-1)*np.conj(p['g'])*Vc**2

        bb_dot = -2*p['gamma_c']*bb + 2*p['N']*np.conj(p['g'])*bVc

        # Package all the time derivatives in a single vector
        z_dot = np.array([
            np.real(b_dot), np.imag(b_dot),
            Bb_dot,
            Cc_dot,
            np.real(Vc_dot), np.imag(Vc_dot),
            np.real(bCv_dot), np.imag(bCv_dot),
            np.real(bCc_dot), np.imag(bCc_dot),
            np.real(bVc_dot), np.imag(bVc_dot),
            np.real(bb_dot), np.imag(bb_dot)
            ])

        return z_dot

class QD_TPM_1F_d(QD):
    """
    A Python object to store and manage data related to the
    Two-Particle Correlation Model of quantum dots with the one
    particle approximation for Fermions and different quantum dots
    (TPM_1F_d). This class does not add any public methods to those of
    the parent class, but overrides the set_parameter method to ensure
    that if the number of quantum dots is changed then all the
    parameters and indices that depend on this value are updated.

    See also: :py:class:`QD.QD`
    
    """

    def __init__(self):
        """
        Initializes the QD_TPM_d object with empty data structures.
        """
        super().__init__()
        # Model type
        self.p['Model'] = "TPM_1F_d"
        # Remove 'Delta_nu' from the list of parameters that the user can change
        self.p['allowed_keys'] = [s for s in self.p['allowed_keys'] if s != 'Delta_nu']
        # Set the parameters that depend on the number of quantum dots
        self.__set_N_dep_params__()
        
    def __set_N_dep_params__(self):
        """
        Function to set the parameters and indices that depend on the number of
        quantum dots.

        Returns
        -------
        None.

        """
        
        # Set the number of equations
        self.p['nf'] = 5 + 9*self.p['N']
        # Define a dictionary of indices that map the physical field to their
        # real and imaginary part format stored in the array z.
        self.q = {}
        # Indices of the various fields in the vector of variables.
        self.q['j_bR'] = 0
        self.q['j_bI'] = 1
        self.q['j_Bb'] = 2
        self.q['j_bbR'] = 3
        self.q['j_bbI'] = 4
        self.q['j_Cc'] = 5 + np.arange(self.p['N'])
        self.q['j_VcR'] = 5 + self.p['N'] + np.arange(self.p['N'])
        self.q['j_VcI'] = 5 + 2*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCvR'] = 5 + 3*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCvI'] = 5 + 4*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCcR'] = 5 + 5*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bCcI'] = 5 + 6*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bVcR'] = 5 + 7*self.p['N'] + np.arange(self.p['N'])
        self.q['j_bVcI'] = 5 + 8*self.p['N'] + np.arange(self.p['N'])

        # Set the indices of the fermionic variables
        self.p['fv'] = np.concatenate([self.q['j_Cc'],self.q['j_VcR'],self.q['j_VcI']])
        # The gain coefficient is now a vector (with all identical values)
        self.p['g_d'] = self.p['g']*np.ones(self.p['N'])
        # Set the detuning distribution 
        if 'Detuning_centre' in self.p.keys():
            # Use the old parameters
            self.set_detuning(self.p['Detuning_centre'],self.p['std_Delta_nu'],self.p['n_outliers'],self.p['rn_seed'])
        else:  
            #use default parameters
            self.set_detuning()
        
    def eq(self,t, z, p, q):
        """
        Returns the time derivative of the Two-Particle Correlation
        Model of quantum dots with the one particle approximation for
        Fermions and different quantum dots (TPM_1F_d), equations
        (S.8) of [3], written in real and imaginary part. The
        equations are written in a frame rotating with the coherent
        field. The detuning is Δν=ν-Δε, with ν the field cavity
        frequency and Δε the quantum dot frequency (difference between
        energy levels).  The variable z passed to this function
        contains either the real and imaginary parts of the complex
        fields, or are the amplitude of the real fields. The mapping
        is as follows:
       
        | b = z[q['j_bR']] + 1j*z[q['j_bI']]
        | Bb = z[q['j_Bb']]
        | Cc = z[q['j_Cc']]
        | Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        | bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]
        | bCc = z[q['j_bCcR']] + 1j*z[q['j_bCcI']]
        | bVc = z[q['j_bVcR']] + 1j*z[q['j_bVcI']]
        | bb = z[q['j_bbR']] + 1j*z[q['j_bbI']]

        where we have used the notation b= :math:`\\langle b \\rangle`
        and B= :math:`\\langle b^{\dagger}\\rangle`, etc.  
    
        Parameters
        ----------
        
        t : float
            Time (unused in this function, but required by ODE solvers).
            
        z : numpy.ndarray
            A 1D numpy array containing the real and imaginary parts of the complex
            fields, or amplitudes of the real fields, stacked.
            
        p : dict
            A dictionary containing the equation coefficients.
            
        q : dictionary
            The indices that map the physical fields to the z array

        Returns
        -------
        
        z_dot : numpy.ndarray
            A 1D numpy array containing the time derivatives of the corresponding
            elements in z, stacked as real and imaginary parts.

        """

        # CIM variables
        b = z[q['j_bR']] + 1j*z[q['j_bI']]
        Bb = z[q['j_Bb']]
        Cc = z[q['j_Cc']]
        Vc = z[q['j_VcR']] + 1j*z[q['j_VcI']]
        bCv = z[q['j_bCvR']] + 1j*z[q['j_bCvI']]

        # Boson-fermion and boson-boson operators
        bCc = z[q['j_bCcR']] + 1j*z[q['j_bCcI']]
        bVc = z[q['j_bVcR']] + 1j*z[q['j_bVcI']]
        bb = z[q['j_bbR']] + 1j*z[q['j_bbI']]

        # The TPM equations for the expectation values

        b_dot = -p['gamma_c']*b + p['g_d'].conj().dot(Vc)

        Bb_dot = -2*p['gamma_c']*Bb + 2*np.real(p['g_d'].dot(bCv))

        Vc_dot = -(p['gamma'] - 1j*p['Delta_nu'])*Vc + p['g_d']*(2*bCc - b)

        Cc_dot = p['r']*(1 - Cc) - (p['gamma_nl'] + p['gamma_nr'])*Cc - 2*np.real(p['g_d']*bCv)

        bCv_dot = -(p['gamma'] + p['gamma_c'] + 1j*p['Delta_nu'])*bCv \
                  + p['g_d'].conj()*(Cc + Bb*(2*Cc - 1) - 4*b*np.conj(b)*Cc + 4*np.real(b*np.conj(bCc))) \
                  + (np.outer(Vc.conj(),Vc) - np.diag(Vc.conj()*Vc)).dot(p['g_d'].conj())

        bCc_dot = -(p['gamma_c'] + p['gamma_nr'])*bCc - p['gamma_nl']*b*Cc \
                  - p['g_d']*(np.conj(Vc)*bb + 2*b*bCv - 2*(b**2)*np.conj(Vc)) \
                  - p['g_d'].conj()*(Vc*Bb + b*np.conj(bCv) - 2*(np.abs(b)**2)*Vc + np.conj(b)*bVc) \
                  + (np.outer(Cc,Vc) - np.diag(Cc*Vc)).dot(p['g_d'].conj()) \
                  + p['r']*b*(1 - Cc)

        bVc_dot = -(p['gamma_c'] + p['gamma'] - 1j*p['Delta_nu'])*bVc \
                  + p['g_d']*(bb*(2*Cc - 1) + 4*b*bCc - 4*Cc*b**2) \
                  + (np.outer(Vc,Vc) - np.diag(Vc*Vc)).dot(p['g_d'].conj())

        bb_dot = -2*p['gamma_c']*bb + 2*p['g_d'].dot(bVc)

        # Package all the time derivatives in a single vector
        z_dot = np.concatenate((
            [np.real(b_dot), np.imag(b_dot), Bb_dot, np.real(bb_dot), np.imag(bb_dot)],
            Cc_dot,
            np.real(Vc_dot), np.imag(Vc_dot),
            np.real(bCv_dot), np.imag(bCv_dot),
            np.real(bCc_dot), np.imag(bCc_dot),
            np.real(bVc_dot), np.imag(bVc_dot)
        ))

        return z_dot
    
    def set_detuning(self,Detuning_centre=0.0,std_Delta_nu=1,n_outliers=5,rn_seed=None):
        """        
        Modification of the QD.set_detuning method that sets also a
        couple of detuning related elements of the parameter
        dictionary p that are specific to the TPM_d equations.
        
        See also: :py:meth:`QD.set_detuning`
        
        """
        super().set_detuning(Detuning_centre,std_Delta_nu,n_outliers,rn_seed)
        self.p['Delta_minus'] = self.p['Delta_nu'][:, np.newaxis] - self.p['Delta_nu'][np.newaxis, :]
        self.p['Delta_plus'] = self.p['Delta_nu'][:, np.newaxis] + self.p['Delta_nu'][np.newaxis, :]

    def set_parameter(self, new_params: dict):
        """
        Sets or updates the values in the parameter dictionary 'p'. It checks
        if the number of quantum dots has changed and regenerates all the 
        parameters and indices that depend on this value.

        Parameters
        ----------
        
            new_params : dictionary
              A dictionary containing new key-value pairs to be added or 
              updated in 'p'. Only names of existing fields are accepted.
        """
        
        # Store the previous value of the number of quantum dots
        N_old = self.p['N']
        # Store the previous value of the gain coefficient
        g_old = self.p['g']
        
        # Update the parameters
        super().set_parameter(new_params)
        # Check if N has changed and update the N-dependent parameters if needed
        if self.p['N'] != N_old:
            self.__set_N_dep_params__()
        # Check if the gain coefficient has changed
        if self.p['g'] != g_old:
            # Generate again the gain coefficients
            self.p['g_d'] = self.p['g']*np.ones(self.p['N'])
         
# Example Usage:
if __name__ == "__main__":
    print ("See docs for examples of usage. Otherwise look at the bottom of the QD.py file.")
    # print()
    # print("*** CIM ***")
    # print()
    # # Create an instance of the QD_CIM object
    # qd1 = QD_CIM()
    # # Change the detuning to be 0.2γ
    # qd1.set_parameter({'Delta_nu' : 0.2*qd1.p['gamma']})
    # # Integrate for 20 units of time with a fine sampling using default
    # # initial conditions (random)
    # qd1.integrate(np.linspace(0,20,201))
    # # Integrate for an additional 10 units of time with a coarser sampling
    # qd1.integrate(np.linspace(0,10,51))
    # # Plot the physical fields
    # qd1.plot_field('b')
    # qd1.plot_field('Bb')
    # qd1.plot_field('Cc')
    # qd1.plot_field('bCv')
    # qd1.plot_field('Vc')
    # # Save the data to a pickle file with default file name
    # qd1.save_to_file()
    
    # # Create a new instance of the QD_CIM object
    # qd2 = QD_CIM()
    # # Assign its values from the file saved by the earlier instance
    # qd2.load_from_file()
    # # Plot a physical field
    # qd2.plot_field('b')
    
    # # Integrate to equilibrium
    # # Create an instance of the QD_CIM object
    # qd3 = QD_CIM()
    # # Change the detuning to be 0.3γ
    # qd3.set_parameter({'Delta_nu' : 0.3*qd3.p['gamma']})
    # # Integrate to equilibirum
    # t_end, z_end = qd3.equilibrium_value()
    # print("Equilibrium has been reached at time: {0:.3f}".format(t_end))
    # # Set the initial condition to the equilibrium value
    # qd3.set_init_cond(qd3.z_eq)
    # # Integrate for 20 units of time with a fine sampling using the 
    # # equilibrium value as initial condition
    # qd3.integrate(np.linspace(0,20,201))
    # # Plot the coherent field at equilibrium
    # qd3.plot_field('b')
       
    # print()
    # print("*** TPM ***")
    # print()
    # # Create an instance of the QD_TPM object
    # qd4 = QD_TPM()
    # # Change the detuning to be 0.2γ
    # qd4.set_parameter({'Delta_nu' : 0.2*qd4.p['gamma']})
    # # Integrate for 20 units of time with a fine sampling using default
    # # initial conditions (random)
    # qd4.integrate(np.linspace(0,20,201))
    # # Integrate for an additional 10 units of time with a coarser sampling
    # qd4.integrate(np.linspace(0,10,51))
    # # Plot the physical fields
    # qd4.plot_field('b')
    # qd4.plot_field('Bb')
    # qd4.plot_field('Vc')
    # qd4.plot_field('CCcc')
    # qd4.plot_field('VVcc')
    # # Save the data to a pickle file with default file name
    # qd4.save_to_file('QD_TPM.pickle')
    
    # # Create a new instance of the QD_TPM object
    # qd5 = QD_TPM()
    # # Assign its values from the file saved by the earlier instance
    # qd5.load_from_file('QD_TPM.pickle')
    # # Plot a physical field
    # qd5.plot_field('VVcc')
    
    # # Integrate to equilibrium
    # # Create an instance of the QD_TPM object
    # qd6 = QD_TPM()
    # # Change the detuning to be 0.3γ
    # qd6.set_parameter({'Delta_nu' : 0.3*qd6.p['gamma']})
    # # Integrate to equilibirum
    # t_end, z_end = qd6.equilibrium_value()
    # print("Equilibrium has been reached at time: {0:.3f}".format(t_end))
    # # Set the initial condition to the equilibrium value
    # qd6.set_init_cond(qd6.z_eq)
    # # Integrate for 20 units of time with a fine sampling using the 
    # # equilibrium value as initial condition
    # qd6.integrate(np.linspace(0,20,401))
    # # Plot the coherent field at equilibrium
    # qd6.plot_field('VVcc')
    
    # print()
    # print("*** TPM_1F ***")
    # print()
    
    # # Create an instance of the QD_TPM_1F object
    # qd7 = QD_TPM_1F()
    # # Change the detuning to be 0.2γ
    # qd7.set_parameter({'Delta_nu' : 0.2*qd7.p['gamma']})
    # # Integrate for 20 units of time with a fine sampling using default
    # # initial conditions (random)
    # qd7.integrate(np.linspace(0,20,201))
    # # Integrate for an additional 10 units of time with a coarser sampling
    # qd7.integrate(np.linspace(0,10,51))
    # # Plot the physical fields
    # qd7.plot_field('b')
    # qd7.plot_field('Bb')
    # qd7.plot_field('Vc')
    # qd7.plot_field('bVc')
    # qd7.plot_field('bb')
    # # Save the data to a pickle file with default file name
    # qd7.save_to_file('QD_TPM_1F.pickle')
    
    # # Create a new instance of the QD_TPM_1F object
    # qd8 = QD_TPM_1F()
    # # Assign its values from the file saved by the earlier instance
    # qd8.load_from_file('QD_TPM_1F.pickle')
    # # Plot a physical field
    # qd8.plot_field('bb')
    
    # # Integrate to equilibrium
    # # Create an instance of the QD_TPM_1F object
    # qd9 = QD_TPM_1F()
    # # Change the detuning to be 0.3γ
    # qd9.set_parameter({'Delta_nu' : 0.3*qd9.p['gamma']})
    # # Integrate to equilibirum
    # t_end, z_end = qd9.equilibrium_value()
    # print("Equilibrium has been reached at time: {0:.3f}".format(t_end))
    # # Set the initial condition to the equilibrium value
    # qd9.set_init_cond(qd9.z_eq)
    # # Integrate for 20 units of time with a fine sampling using the 
    # # equilibrium value as initial condition
    # qd9.integrate(np.linspace(0,20,401))
    # # Plot the coherent field at equilibrium
    # qd9.plot_field('bb')
    
    # print()
    # print("*** CIM_d ***")
    # print()
    # # Create an instance of the QD_CIM object
    # qd1d = QD_CIM_d()
    # # Increase the number of quantum dots
    # qd1d.set_parameter({'N':40})
    # # Change the detuning to be 0.2γ and standard deviation γ with 4 outliers
    # # qd1d.set_detuning(0.2,1,4,None)
    # # Integrate for 20 units of time with a fine sampling using default
    # # initial conditions (random)
    # qd1d.integrate(np.linspace(0,20,201))
    # # # Integrate for an additional 10 units of time with a coarser sampling
    # # qd1d.integrate(np.linspace(0,10,51))
    # # # Plot the physical fields
    # qd1d.plot_field('b')
    # qd1d.plot_field('Bb')
    # qd1d.plot_field('Cc',np.array([4,21,38]))
    # qd1d.plot_field('bCv',np.array([4,21,38]))
    # qd1d.plot_field('Vc',np.array([4,21,38]))
    # # Save the data to a pickle file with default file name
    # qd1d.save_to_file('QD_CIM_d.pickle')
    
    # # Create a new instance of the QD_CIM_d object
    # qd2d = QD_CIM_d()
    # # Assign its values from the file saved by the earlier instance
    # qd2d.load_from_file('QD_CIM_d.pickle')
    # # Plot a physical field
    # qd2d.plot_field('b')
    
    # # Integrate to equilibrium
    # # Create an instance of the QD_CIM_d object
    # qd3d = QD_CIM_d()
    # # Change the detuning to be 0.3γ
    # qd3d.set_parameter({'Delta_nu' : 0.3*qd3d.p['gamma']})
    # # Increase the number of quantum dots
    # qd3d.set_parameter({'N':40})
    # # Integrate to equilibirum
    # t_end, z_end = qd3d.equilibrium_value()
    # print("Equilibrium has been reached at time: {0:.3f}".format(t_end))
    # # Set the initial condition to the equilibrium value
    # qd3d.set_init_cond(qd3d.z_eq)
    # # Integrate for 20 units of time with a fine sampling using the 
    # # equilibrium value as initial condition
    # qd3d.integrate(np.linspace(0,20,201))
    # # Plot the coherent field at equilibrium
    # qd3d.plot_field('b')    
         
    # print()
    # print("*** TPM_d ***")
    # print("Note that the integration time can be very long: 2 hours for 2 time units!")
    # print()
    # # Create an instance of the QD_TPM_d object
    # qd4d = QD_TPM_d()
    # # Increase the number of quantum dots
    # qd4d.set_parameter({'N':50})
    # # Change the detuning to be 0.2γ and standard deviation γ with 4 outliers
    # qd4d.set_detuning(0.2,1,4,None)
    # # Integrate for 2 units of time with a fine sampling using default
    # # initial conditions (random)
    # qd4d.integrate(np.linspace(0,2,21))
    # # Integrate for an additional 10 units of time with a coarser sampling
    # # qd4d.integrate(np.linspace(0,10,51))
    # # Plot the physical fields
    # qd4d.plot_field('b')
    # qd4d.plot_field('Bb')
    # qd4d.plot_field('Vc',np.array([4,21,38]))
    # qd4d.plot_field('CCcc',np.array([[0,14],[28,2],[4,15]]))
    # qd4d.plot_field('VCcc',np.array([[0,14],[28,2],[4,15]]))
    # # qd4d.plot_field('CCcc')
    # # qd4d.plot_field('VVcc')
    # # Save the data to a pickle file with default file name
    # qd4d.save_to_file('QD_TPM_d.pickle')
    
    # Create a new instance of the QD_TPM_d object
    # qd5d = QD_TPM_d()
    # # Assign its values from the file saved by the earlier instance
    # qd5d.load_from_file('QD_TPM_d.pickle')
    # # Map the variables
    # qd5d.field_mapping()
    # # Obtains some field values
    # b = qd5d.field_values('b')
    # CCcc = qd5d.field_values('CCcc',np.array([[0,14],[28,2],[4,15]]))
    # # Plot a physical field
    # qd5d.plot_field('b')
    
    # # Integrate to equilibrium
    # # Create an instance of the QD_TPM_d object
    # qd6d = QD_TPM_d()
    # # Change the detuning to be 0.3γ
    # qd6d.set_parameter({'Delta_nu' : 0.3*qd6d.p['gamma']})
    # # Integrate to equilibirum
    # t_end, z_end = qd6d.equilibrium_value()
    # print("Equilibrium has been reached at time: {0:.3f}".format(t_end))
    # # Set the initial condition to the equilibrium value
    # qd6d.set_init_cond(qd6d.z_eq)
    # # Integrate for 20 units of time with a fine sampling using the 
    # # equilibrium value as initial condition
    # qd6d.integrate(np.linspace(0,20,401))
    # # Plot the coherent field at equilibrium
    # qd6d.plot_field('VVcc')  
    
    # print()
    # print("*** TPM_1F_d ***")
    # print()
    # # Create an instance of the QD_CIM object
    # qd7d = QD_TPM_1F_d()
    # # Increase the number of quantum dots
    # qd7d.set_parameter({'N':40})
    # # Change the detuning to be 0.2γ and standard deviation γ with 4 outliers
    # qd7d.set_detuning(0.2,1,4,None)
    # # Integrate for 20 units of time with a fine sampling using default
    # # initial conditions (random)
    # qd7d.integrate(np.linspace(0,20,201))
    # # # Integrate for an additional 10 units of time with a coarser sampling
    # qd7d.integrate(np.linspace(0,10,51))
    # # # Plot the physical fields
    # qd7d.plot_field('b')
    # qd7d.plot_field('Bb')
    # qd7d.plot_field('Cc',np.array([4,21,38]))
    # qd7d.plot_field('bCv',np.array([4,21,38]))
    # qd7d.plot_field('Vc',np.array([4,21,38]))
    # # Save the data to a pickle file with default file name
    # qd7d.save_to_file('QD_TPM_1F_d.pickle')
    
    # # Create a new instance of the QD_TPM_1F_d object
    # qd8d = QD_TPM_1F_d()
    # # Assign its values from the file saved by the earlier instance
    # qd8d.load_from_file('QD_TPM_1F_d.pickle')
    # # Plot a physical field
    # qd8d.plot_field('b')
    
    # # Integrate to equilibrium
    # # Create an instance of the QD_TPM_1F_d object
    # qd9d = QD_TPM_1F_d()
    # # Change the detuning to be 0.3γ
    # qd9d.set_parameter({'Delta_nu' : 0.3*qd9d.p['gamma']})
    # # Increase the number of quantum dots
    # qd9d.set_parameter({'N':50})
    # # Integrate to equilibirum
    # t_end, z_end = qd9d.equilibrium_value()
    # print("Equilibrium has been reached at time: {0:.3f}".format(t_end))
    # # Set the initial condition to the equilibrium value
    # qd9d.set_init_cond(qd9d.z_eq)
    # # Integrate for 20 units of time with a fine sampling using the 
    # # equilibrium value as initial condition
    # qd9d.integrate(np.linspace(0,20,201))
    # # Plot the coherent field at equilibrium
    # qd9d.plot_field('b')  
