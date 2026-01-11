'''
	callbacks.py - v2020.05.12

	Authors: Andreas Tritsarolis, Christos Doulkeridis, Yannis Theodoridis and Nikos Pelekis
'''    


import abc
import bokeh.models as bokeh_mdl


class BokehFilters:
    __metaclass__ = abc.ABCMeta
    """
    Abstract base class for interactive filter widgets in VISIONS instances
    
    Provides a framework for connecting Bokeh widgets to data filtering operations
    in ST_Visions visualizations. Handles callback coordination, data state management,
    and synchronized updates across multiple filtering widgets.
    
    Parameters
    ----------
    vsn_instance : st_visualizer
        The ST_Visions visualization instance that filters will be applied to.
    widget : bokeh.models.Widget
        Bokeh widget (slider, select, etc.) that triggers filtering callbacks.
    
    Attributes
    ----------
    widget : bokeh.models.Widget
        The widget associated with this filter instance.
    vsn_instance : st_visualizer
        Reference to the parent visualization instance containing data and state.
    
    Notes
    -----
    This class uses a locking mechanism (`aquire_canvas_data`) to coordinate
    multiple filter widgets and prevent race conditions during data updates.

    Subclasses must implement the `callback` method to define specific filter logic.
    """

    def __init__(self, vsn_instance, widget):
        '''
        Initialize BokehFilters instance.
        
        Parameters
        ----------
        vsn_instance : st_visualizer
            The VISIONS instance that the BokehFilters instance will be connected to.
        widget : bokeh.models.Widget
            The widget that the callback will be intended for.
        '''
        self.widget = widget
        self.vsn_instance = vsn_instance


    def callback_filter_data(self):
        '''
        Iteratively triggers the widgets' callback methods in order to filter the data. 

        Notes
        -----
        This method should be called **first** in any custom callback implementation.
        '''
        if not self.vsn_instance.aquire_canvas_data:
            self.vsn_instance.aquire_canvas_data = self.widget.id
            
            for widget in self.vsn_instance.widgets:
                if not widget.id == self.widget.id:
                    widget_callback_policy = list(widget._callbacks.keys())[0] 
                    widget.trigger(widget_callback_policy, None, widget.value)


    def get_data(self):
        '''
        Retrieve the appropriate dataset for filtering operations.
        
        Notes
        -----
         If the lock is aquired:
          * If the intermediate storage (canvas_data) is empty fetch the loaded dataset; otherwise
          * Fetch the filtered data via the intermediate storage.
        '''
        if self.vsn_instance.aquire_canvas_data and (self.vsn_instance.canvas_data is not None):        
            # print ('Fetching Filtered Data')
            return self.vsn_instance.canvas_data
        else:
            # print ('Fetching OG Data')
            return self.vsn_instance.data


    def callback_prepare_data(self, new_pts, ready_for_output):
        '''
        Preparing the Filtered data prior to rendering (i.e., passing them to the CDS). 
        
        This method is recommended to be placed last in a custom callback method
        '''
        self.vsn_instance.canvas_data = new_pts

        if ready_for_output:
            self.vsn_instance.canvas_data = self.vsn_instance.prepare_data(self.vsn_instance.canvas_data)

            if (self.vsn_instance.cmap is not None) and (isinstance(self.vsn_instance.cmap['transform'], bokeh_mdl.CategoricalColorMapper)):
                factors = sorted(self.vsn_instance.canvas_data[self.vsn_instance.cmap['field']].unique().tolist())
                self.vsn_instance.cmap['transform'].factors = factors

            self.vsn_instance.source.data = self.vsn_instance.canvas_data.drop(self.vsn_instance.canvas_data.geometry.name, axis=1).to_dict(orient="list")

            # print ('Releasing Lock...')
            self.vsn_instance.canvas_data = None
            self.vsn_instance.aquire_canvas_data = None


    @abc.abstractmethod
    def callback(self, *args):
        pass