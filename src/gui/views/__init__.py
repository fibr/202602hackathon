"""View registry for the unified GUI.

Each view is a subclass of BaseView. Views are auto-discovered from this
package and registered by their `view_id` class attribute.
"""

from gui.views.base import BaseView, ViewRegistry

__all__ = ['BaseView', 'ViewRegistry']
