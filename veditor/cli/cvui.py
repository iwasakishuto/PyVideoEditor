"""
A (very) simple UI lib built on top of OpenCV drawing primitives.
Version: 2.7

Use of ``cvui`` revolves around calling ``cvui.init()`` to initialize the lib, rendering cvui components to a np.ndarray (that you handle yourself) and finally showing that np.ndarray on the screen using cvui.imshow(), which is cvui's version of cv2.imshow(). Alternatively you can use cv2.imshow() to show things, but in such case you must call ``cvui.update()`` yourself before calling cv.imshow().

Read the full documentation at `cvui - (very) simple UI lib built on top of OpenCV drawing primitives <https://dovyski.github.io/cvui/>`_

Copyright (c) 2018 `Fernando Bevilacqua <https://github.com/Dovyski>`_
Licensed under the `MIT license <https://github.com/Dovyski/cvui/blob/master/LICENSE.md>`_.
"""
import sys

import cv2
import numpy as np

from ..utils._colorings import toRED
from ..utils.color_utils import choose_text_color, generate_color_series
from ..utils.generic_utils import NoneType, handleTypeError, now_str

# OpenCV color
cv2BLACK = (0, 0, 0)
cv2RED = (0, 0, 255)
cv2GREEN = (0, 128, 0)
cv2YELLOW = (0, 255, 255)
cv2BLUE = (255, 0, 0)
cv2MAGENTA = (255, 0, 255)
cv2CYAN = (255, 255, 0)
cv2WHITE = (255, 255, 255)

# Constants regarding component interactions
ROW = 0
COLUMN = 1
DOWN = 2
CLICK = 3
OVER = 4
OUT = 5
UP = 6
IS_DOWN = 7

# OpenCV Key.
ESCAPE = 27

# Constants regarding mouse buttons
LEFT_BUTTON = 0
MIDDLE_BUTTON = 1
RIGHT_BUTTON = 2

# Constants regarding components
TRACKBAR_HIDE_SEGMENT_LABELS = 1
TRACKBAR_HIDE_STEP_SCALE = 2
TRACKBAR_DISCRETE = 4
TRACKBAR_HIDE_MIN_MAX_LABELS = 8
TRACKBAR_HIDE_VALUE_LABEL = 16
TRACKBAR_HIDE_LABELS = 32

# Internal things
CVUI_ANTIALISED = cv2.LINE_AA
CVUI_FILLED = -1


def init(windowNames=now_str(), numWindows=1, delayWaitKey=-1, createNamedWindows=True):
    """Initialize cvui using a list of names of windows where components will be added.

    It is also possible to tell cvui to handle OpenCV's event queue automatically (by informing a value greater than zero in the `delayWaitKey` parameter of the function).

    In that case, cvui will automatically call ``cv2.waitKey()`` within ``cvui.update()`` , so you don't have to worry about it. The value passed to `delayWaitKey` will be used as the delay for `cv2.waitKey()`.

    Args:
            windowNames (str,list)    : Array containing the name of the windows where components will be added. Those windows will be automatically if ``createNamedWindows`` is ``True`` .
            numWindows (int)          : How many window names exist in the ``windowNames`` array.
            delayWaitKey (int)        : Delay value passed to ``cv2.waitKey()``. If a negative value is informed (default is ``-1``), cvui will not automatically call ``cv2.waitKey()`` within ``cvui.update()``, which will disable keyboard shortcuts for all components. If you want to enable keyboard shortcut for components (e.g. using & in a button label), you must specify a positive value for this param.
            createNamedWindows (bool) : If OpenCV windows named according to `windowNames` should be created during the initialization. Windows are created using ``cv2.namedWindow()`` . If this parameter is ``False`` , ensure you call ``cv2.namedWindow(WINDOW_NAME)`` for all windows *before* initializing cvui, otherwise it will not be able to track UI interactions.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            >>> from veditor.__meta__ import __version__
            ...
            >>> WINDOW_NAME	= "CVUI Test"
            >>> frame = np.zeros((300, 600, 3), np.uint8)
            >>> checked = [False]
            >>> checked2 = [True]
            >>> count = [0]
            >>> countFloat = [0.0]
            >>> trackbarValue = [0.0]
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	# Show some pieces of text.
            ... 	cvui.text(frame, 50, 30, "Hey there!")
            ...
            ... 	# You can also specify the size of the text and its color
            ... 	# using hex 0xRRGGBB CSS-like style.
            ... 	cvui.text(frame, 200, 30, "Use hex 0xRRGGBB colors easily", fontScale=0.4, color=0xff0000)
            ...
            ... 	# Sometimes you want to show text that is not that simple, e.g. strings + numbers.
            ... 	# You can use cvui.printf for that. It accepts a variable number of parameter, pretty
            ... 	# much like printf does.
            ... 	cvui.printf(frame, 200, 50, fontScale=0.4, color=0x00ff00, fmt="Use printf formatting: %d + %.2f = %f", fmtArgs=(2, 3.2, 5.2))
            ...
            ... 	# Buttons will return true if they were clicked, which makes
            ... 	# handling clicks a breeze.
            ... 	if cvui.button(frame, 50, 60, "Button"):
            ... 		print("Button clicked")
            ...
            ... 	# If you do not specify the button width/height, the size will be
            ... 	# automatically adjusted to properly house the label.
            ... 	cvui.button(frame, 200, 70, "Button with large label")
            ...
            ... 	# You can tell the width and height you want
            ... 	cvui.button(frame, 410, 70, "x", 15, 15)
            ...
            ... 	# Window components are useful to create HUDs and similars. At the
            ... 	# moment, there is no implementation to constraint content within a
            ... 	# a window.
            ... 	cvui.window(frame, 50, 120, 120, 100, "Window")
            ...
            ... 	# The counter component can be used to alter int variables. Use
            ... 	# the 4th parameter of the function to point it to the variable
            ... 	# to be changed.
            ... 	cvui.counter(frame, 200, 120, count)
            ...
            ... 	# Counter can be used with doubles too. You can also specify
            ... 	# the counter's step (how much it should change
            ... 	# its value after each button press), as well as the format
            ... 	# used to print the value.
            ... 	cvui.counter(frame, 320, 120, countFloat, 0.1, "%.1f")
            ...
            ... 	# The trackbar component can be used to create scales.
            ... 	# It works with all numerical types (including chars).
            ... 	cvui.trackbar(frame, 420, 110, 150, trackbarValue, 0., 50.)
            ...
            ... 	# Checkboxes also accept a pointer to a variable that controls
            ... 	# the state of the checkbox (checked or not). cvui.checkbox() will
            ... 	# automatically update the value of the boolean after all
            ... 	# interactions, but you can also change it by yourself. Just
            ... 	# do "checked = [True]" somewhere and the checkbox will change
            ... 	# its appearance.
            ... 	cvui.checkbox(frame, 200, 160, "Checkbox", checked)
            ... 	cvui.checkbox(frame, 200, 190, "A checked checkbox", checked2)
            ...
            ... 	# Display the lib version at the bottom of the screen
            ... 	cvui.printf(frame, 600-150, 300-20, fontScale=0.4, color=0xCECECE, fmt="veditor v.%s", fmtArgs=(__version__))
            ...
            ... 	# This function must be called *AFTER* all UI components. It does
            ... 	# all the behind the scenes magic to handle mouse clicks, etc.
            ... 	cvui.update()
            ...
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ...
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    if isinstance(windowNames, str):
        windowNames = [windowNames]
        numWindows = 1
    else:
        numWindows = len(windowNames)

    __internal.init(windowNames[0], delayWaitKey)
    for windowName in windowNames:
        watch(windowName, createNamedWindows)


def _handleMouse(event, x, y, flags, context):
    Buttons = [LEFT_BUTTON, MIDDLE_BUTTON, RIGHT_BUTTON]
    EventsDown = [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]
    EventsUp = [cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP]

    for i, btn in enumerate(Buttons):
        if event == EventsDown[i]:
            context.mouse.anyButton.justPressed = True
            context.mouse.anyButton.pressed = True
            context.mouse.buttons[btn].justPressed = True
            context.mouse.buttons[btn].pressed = True
        elif event == EventsUp[i]:
            context.mouse.anyButton.justReleased = True
            context.mouse.anyButton.pressed = False
            context.mouse.buttons[btn].justReleased = True
            context.mouse.buttons[btn].pressed = False

    context.mouse.position.x = x
    context.mouse.position.y = y


def watch(windowName, createNamedWindow=True):
    """Track UI interactions of a particular window. This function must be invoked for any window that will receive cvui components. cvui automatically calls ``cvui.watch()`` for any window informed in `cvui.init()`, so generally you don't have to watch them yourself. If you initialized cvui and told it *not* to create windows automatically, you need to call ``cvui.watch()`` on those windows yourself. ``cvui.watch()`` can automatically create a window before watching it, if it does not exist.

    Args:
            windowName (str)         : Name of the window whose UI interactions will be tracked.
            createNamedWindow (bool) : If an OpenCV window named ``windowName`` should be created before it is watched. Windows are created using ``cv2.namedWindow()``. If this parameter is ``False``, ensure you have called ``cv2.namedWindow(WINDOW_NAME)`` to create the window, otherwise cvui will not be able to track its UI interactions.
    """
    if createNamedWindow:
        cv2.namedWindow(windowName)

    context = Context()
    context.windowName = windowName
    context.mouse.position.x, context.mouse.position.y = (0, 0)

    context.mouse.anyButton.reset()
    context.mouse.buttons[RIGHT_BUTTON].reset()
    context.mouse.buttons[MIDDLE_BUTTON].reset()
    context.mouse.buttons[LEFT_BUTTON].reset()

    __internal.contexts[windowName] = context
    cv2.setMouseCallback(window_name=windowName, on_mouse=_handleMouse, param=context)


def context(windowName):
    """
    Inform cvui that all subsequent component calls belong to a window in particular. When using cvui with multiple OpenCV windows, you must call cvui component calls between ``cvui.contex(NAME)`` and ``cvui.update(NAME)``, where ``NAME`` is the name of the window. That way, cvui knows which window you are using ( ``NAME`` in this case), so it can track mouse events, for instance.

    Pay attention to the pair ``cvui.context(NAME)`` and ``cvui.update(NAME)`` , which encloses the component calls for that window. You need such pair for each window of your application.

    After calling ``cvui.update()``, you can show the result in a window using ``cv2.imshow()``. If you want to save some typing, you can use ``cvui.imshow()``, which calls ``cvui.update()`` for you and then shows the frame in a window.

    In that case, you don't have to bother calling ``cvui.update()`` yourself, since ``cvui.imshow()`` will do it for you.

    Args:
            windowName (str) : Name of the window that will receive components from all subsequent cvui calls.
    """
    __internal.currentContext = windowName


def imshow(windowName, frame):
    """
    Display an image in the specified window and update the internal structures of cvui. This function can be used as a replacement for ``cv2.imshow()``. If you want to use ``cv2.imshow()`` instead of ``cvui.imshow()``, you must ensure you call ``cvui.update()`` *after* all component calls and *before* ``cv2.imshow()``, so cvui can update its internal structures.

    In general, it is easier to call ``cvui.imshow()`` alone instead of calling ``cvui.update()`` immediately followed by ``cv2.imshow()``.

    Args:
            windowName (str)   : Name of the window that will be shown.
            frame (np.ndarray) : Image, i.e. ``np.ndarray``, to be shown in the window.
    """
    update(windowName)
    cv2.imshow(windowName, frame)


def lastKeyPressed():
    """Return the last key that was pressed. This function will only work if a value greater than zero was passed to ``cvui.init()`` as the delay waitkey parameter."""
    return __internal.lastKeyPressed


def iarea(x, y, width, height):
    """
    Create an interaction area that reports activity with the mouse cursor. The tracked interactions are returned by the function and they are:

    - ``cvui.OUT`` when the cursor is not over the iarea.
    - ``cvui.OVER`` when the cursor is over the iarea.
    - ``cvui.DOWN`` when the cursor is pressed over the iarea, but not released yet.
    - ``cvui.CLICK`` when the cursor clicked (pressed and released) within the iarea.

    This function creates no visual output on the screen. It is intended to be used as an auxiliary tool to create interactions.

    Args:
            x (int)      : Position X where the interactive area should be placed.
            y (int)      : Position Y where the interactive area should be placed.
            width (int)  : Width of the interactive area.
            height (int) : Height of the interactive area.

    Returns:
            int : An integer value representing the current state of interaction with the mouse cursor. It can be
                                            - ``cvui.OUT`` when the cursor is not over the iarea.
                                            - ``cvui.OVER`` when the cursor is over the iarea.
                                            - ``cvui.DOWN`` when the cursor is pressed over the iarea, but not released yet.
                                            - ``cvui.CLICK`` when the cursor clicked (pressed and released) within the iarea.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= 'Interaction area'
            >>> frame = np.zeros((300, 600, 3), np.uint8)
            >>> cvui.init(WINDOW_NAME)
            >>> rectangle = cvui.Rect(50, 50, 100, 100)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	# Render a rectangle on the screen.
            ... 	cvui.rect(frame, rectangle.x, rectangle.y, rectangle.width, rectangle.height, 0xff0000)
            ...
            ... 	# Check what is the current status of the mouse cursor
            ... 	# regarding the previously rendered rectangle.
            ... 	status = cvui.iarea(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
            ...
            ... 	# cvui.iarea() will return the current mouse status:
            ... 	#  CLICK : mouse just clicked the interaction are
            ... 	#  DOWN  : mouse button was pressed on the interaction area, but not released yet.
            ... 	#  OVER  : mouse cursor is over the interaction area
            ... 	#  OUT   : mouse cursor is outside the interaction area
            ... 	if status == cvui.CLICK:  print('Rectangle was clicked!')
            ... 	if status == cvui.DOWN:   cvui.text(frame, 240, 70, text="Mouse is: DOWN")
            ... 	if status == cvui.OVER:   cvui.text(frame, 240, 70, text="Mouse is: OVER")
            ... 	if status == cvui.OUT:    cvui.text(frame, 240, 70, text="Mouse is: OUT")
            ...
            ... 	# Show the coordinates of the mouse pointer on the screen
            ... 	cvui.printf(frame, 240, 50, fmt="Mouse pointer is at (%d,%d)", fmtArgs=(cvui.mouse().x, cvui.mouse().y))
            ...
            ... 	# This function must be called *AFTER* all UI components. It does
            ... 	# all the behind the scenes magic to handle mouse clicks, etc.
            ... 	cvui.update()
            ...
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ...
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    return __internal.iarea(x, y, width, height)


def space(value=5):
    """
    Add an arbitrary amount of space between components within a ``begin*()`` and ``end*()`` block.
    The function is aware of context, so if it is used within a `beginColumn()` and
    `endColumn()` block, the space will be vertical. If it is used within a `beginRow()`
    and `endRow()` block, space will be horizontal.

    NOTE: This function can only be used within a ``begin*()/end*()`` block, otherwise it does nothing.

    Args:
            value (int) : The amount of space to be added.
    """
    block = __internal.topBlock()
    size = Size(value, value)

    __internal.updateLayoutFlow(block, size)


def update(windowName=""):
    """Update the library internal things. You need to call this function **AFTER** you are done adding/manipulating UI elements in order for them to react to mouse interactions.

    Args:
            windowName (str) : Name of the window whose components are being updated. If no window name is provided, cvui uses the default window.
    """
    context = __internal.getContext(windowName)

    context.mouse.anyButton.justReleased = False
    context.mouse.anyButton.justPressed = False

    for i in range(LEFT_BUTTON, RIGHT_BUTTON + 1):
        context.mouse.buttons[i].justReleased = False
        context.mouse.buttons[i].justPressed = False

    __internal.screen.reset()

    # If we were told to keep track of the keyboard shortcuts, we
    # proceed to handle opencv event queue.
    if __internal.delayWaitKey > 0:
        __internal.lastKeyPressed = cv2.waitKey(__internal.delayWaitKey)

    if not __internal.isblockStackEmpty:
        __internal.error(
            2,
            "Calling update() before finishing all begin*()/end*() calls. Did you forget to call a begin*() or an end*()? Check if every begin*() has an appropriate end*() call before you call update().",
        )


def text(
    where=None,
    x=0,
    y=0,
    text="",
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.4,
    color=0xCECECE,
    thickness=1,
    lineType=cv2.LINE_8,
):
    """Display a piece of text.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            text (str)         : The text content.
            fontFace (int)     : Font type. (default= ``cv2.FONT_HERSHEY_SIMPLEX`` )
            fontScale (float)  : Font scale factor that is multiplied by the font-specific base size.
            color (uint)       : Color of the text in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
            thickness (int)    : Thickness of the lines used to draw a text.
            lineType (int)     : Line type. (default= ``cv2.LINE_8`` )

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= "Text"
            >>> frame = np.zeros(shape=(50, 100, 3), dtype=np.uint8)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	frame[:] = (49, 52, 49)
            ... 	cvui.text(where=frame, x=10, y=20, text="Hey there!")
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    __internal.text(
        block,
        x,
        y,
        text,
        fontFace,
        fontScale,
        color,
        thickness,
        lineType,
        updateLayout=True,
    )


def printf(
    where=None,
    x=0,
    y=0,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.4,
    color=0xCECECE,
    thickness=1,
    lineType=cv2.LINE_8,
    fmt="Text: %s",
    fmtArgs=(""),
):
    """Display a piece of text that can be formated using ``C stdio's printf()`` style. For instance if you want to display text mixed with numbers, you can use:

    .. code-block:: python

            >>> print("Text: %d and %f" % (7, 3.1415))

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            fontFace (int)     : Font type. (default= ``cv2.FONT_HERSHEY_SIMPLEX`` )
            fontScale (float)  : Font scale factor that is multiplied by the font-specific base size.
            color (uint)       : Color of the text in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
            thickness (int)    : Thickness of the lines used to draw a text.
            lineType (int)     : Line type. (default= ``cv2.LINE_8`` )
            fmt (str)          : Formating string as it would be supplied for stdio's ``printf()``, e.g. ``'Text: %d and %f', 7, 3.1415``.
            fmtArgs (tuple)    : Arguments for ``fmt`` .

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            >>> from veditor.utils import now_str
            ...
            >>> WINDOW_NAME	= 'Printf'
            >>> frame = np.zeros(shape=(50, 250, 3), dtype=np.uint8)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ...  	frame[:] = (49, 52, 49)
            ...  	cvui.printf(frame, 20, 20, fontScale=0.4, color=0xCECECE, fmt="Date: %s", fmtArgs=(now_str()))
            ...  	cvui.update()
            ...  	cv2.imshow(WINDOW_NAME, frame)
            ...  	if cv2.waitKey(20) == cvui.ESCAPE:
            ...  		break
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    text = fmt % fmtArgs
    __internal.text(
        block,
        x,
        y,
        text,
        fontFace,
        fontScale,
        color,
        thickness,
        lineType,
        updateLayout=True,
    )


def counter(where=None, x=0, y=0, value=[], step=1, fmt=""):
    """Display a counter for integer values that the user can increase/descrease by clicking the up and down arrows.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            value ([number])   : Array or List of numbers whose first position, i.e. ``value[0]``, will be used to store the current value of the counter.
            step (number)      : Amount that should be increased/decreased when the user interacts with the counter buttons
            fmt (str)          : How the value of the counter should be presented, as it was printed by ``stdio's printf()``. E.g. ``'%d'`` means the value will be displayed as an integer, ``'%0d'`` integer with one leading zero, etc.

    Returns:
            value (number) : Number that corresponds to the current value of the counter.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            >>> from veditor.utils import now_str
            ...
            >>> WINDOW_NAME	= 'Text'
            >>> frame = np.zeros(shape=(100, 150, 3), dtype=np.uint8)
            >>> cvui.init(WINDOW_NAME)
            >>> countFloat = [0.]
            ...
            >>> while (True):
            ... 	frame[:] = (49, 52, 49)
            ... 	cvui.counter(frame, 10, 10, countFloat, 0.1, '%.1f')
            ... 	cvui.printf(frame,  10, 50, fontScale=0.4, color=0xCECECE, fmt="Current value: %.1f", fmtArgs=(countFloat[0]))
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    if len(fmt) == 0:
        fmt = "%d" if (isinstance(value[0], int) and isinstance(step, int)) else "%.1f"

    return __internal.counter(block, x, y, value, step, fmt)


def checkbox(where=None, x=0, y=0, label="", state=[], color=0xCECECE):
    """Display a checkbox. You can use the state parameter to monitor if the checkbox is checked or not.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            label (str)        : Text displayed besides the clickable checkbox square.
            state ([bool])     : Array or List of booleans whose first position, i.e. ``state[0]``, will be used to store the current state of the checkbox: ``True`` means the checkbox is checked.
            color (uint)       : Color of the label in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.

    Returns:
            value (bool)       : Whether the current state of the checkbox, ``True`` if it is checked.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            >>> from veditor.utils import now_str
            ...
            >>> WINDOW_NAME	= 'Check Box'
            >>> frame = np.zeros(shape=(100, 150, 3), dtype=np.uint8)
            >>> cvui.init(WINDOW_NAME)
            >>> checked = [True]
            ...
            >>> while (True):
            ... 	frame[:] = (49, 52, 49)
            ... 	cvui.checkbox(frame, 10, 10, 'Checkbox', checked)
            ... 	cvui.printf(frame,  10, 50, fontScale=0.4, color=0xCECECE, fmt="Current value: %s", fmtArgs=(checked[0]))
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)

    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    return __internal.checkbox(block, x, y, label, state, color)


def radiobox(where=None, x=0, y=0, labels=[""], states=[], color=0xCECECE):
    """Display a radiobox. You can use the ``states`` parameter to check which radio button is checked.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            labels ([str])     : Text displayed besides the each clickable radiobox square.
            states ([bool])    : Array or List of booleans whose first position, i.e. ``state[0]``, will be used to store the current state of the checkbox: ``True`` means the checkbox is checked.
            color (uint)       : Color of the label in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.

    Returns:
            value (int)        : Which radio button is checked ( ``True`` ).

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui, cv2RED, cv2BLUE, cv2GREEN
            >>> from veditor.utils import now_str
            ...
            >>> WINDOW_NAME	= 'Radio Box'
            >>> frame = np.zeros(shape=(150, 200, 3), dtype=np.uint8)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> radio_labels = ["red", "blue", "green"]
            >>> radio_states = [True, False, False]
            >>> bg_colors    = [cv2RED, cv2BLUE, cv2GREEN]
            >>> idx = 0
            ...
            >>> while (True):
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	cvui.beginRow(frame, x=10, y=10, width=180, height=130, bgColor=bg_colors[idx])
            ... 	idx = cvui.radiobox(x=10, y=10, labels=radio_labels, states=radio_states)
            ... 	cvui.printf(where=frame, x=10, y=100, fontScale=0.4, color=0xCECECE, fmt="Current bgColor: %s", fmtArgs=(radio_labels[idx]))
            ... 	cvui.endRow()
            ...
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y
    return __internal.radiobox(block, x, y, labels, states, color)


def mouse(windowName=None, button=None, query=None):
    """Query the mouse for events in a particular button in a particular window. This function behave exactly like ``mouse(int button, int query)``, with the difference that queries are targeted at  particular mouse button in a particular window instead.

    Args:
            windowName (str) : Name of the window that will be queried.
            button (int)     : An integer describing the mouse button to be queried. Possible values are ``cvui.LEFT_BUTTON``, ``cvui.MIDDLE_BUTTON`` and ``cvui.LEFT_BUTTON``.
            query (int)      : An integer describing the intended mouse query. Available queries are ``cvui.DOWN``, ``cvui.UP``, ``cvui.CLICK``, and ``cvui.IS_DOWN``.

    Returns:
            point (Point)        : (If ``query`` is None)  A point containing the position of the mouse cursor in the speficied window.
            isMouseButton (bool) : (otherwise) Whether the ``button`` and ``query`` are match.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= "Mouse"
            >>> frame = np.zeros((300, 600, 3), np.uint8)
            >>> cvui.init(WINDOW_NAME)
            >>> rectangle = cvui.Rect(0, 0, 0, 0)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	# Show the coordinates of the mouse pointer on the screen
            ... 	cvui.text(frame, 10, 30, 'Click (any) mouse button and drag the pointer around to select an area.')
            ... 	cvui.printf(frame, 10, 50, fontScale=0.4, color=0xff0000, fmt='Mouse pointer is at (%d,%d)', fmtArgs=(cvui.mouse().x, cvui.mouse().y))
            ...
            ... 	# The function "bool cvui.mouse(int query)" allows you to query the mouse for events.
            ... 	# E.g. cvui.mouse(cvui.DOWN)
            ... 	#
            ... 	# Available queries:
            ... 	#	- cvui.DOWN: any mouse button was pressed. cvui.mouse() returns true for a single frame only.
            ... 	#	- cvui.UP: any mouse button was released. cvui.mouse() returns true for a single frame only.
            ... 	#	- cvui.CLICK: any mouse button was clicked (went down then up, no matter the amount of frames in between). cvui.mouse() returns true for a single frame only.
            ... 	#	- cvui.IS_DOWN: any mouse button is currently pressed. cvui.mouse() returns true for as long as the button is down/pressed.
            ...
            ... 	# Did any mouse button go down?
            ... 	if cvui.mouse(query=cvui.DOWN):
            ... 		# Position the rectangle at the mouse pointer.
            ... 		rectangle.x = cvui.mouse().x
            ... 		rectangle.y = cvui.mouse().y
            ...
            ... 	# Is any mouse button down (pressed)?
            ... 	if cvui.mouse(query=cvui.IS_DOWN):
            ... 		# Adjust rectangle dimensions according to mouse pointer
            ... 		rectangle.width = cvui.mouse().x - rectangle.x
            ... 		rectangle.height = cvui.mouse().y - rectangle.y
            ...
            ... 		# Show the rectangle coordinates and size
            ... 		cvui.printf(frame, rectangle.x    + 5, rectangle.y    + 5, fontScale=0.3, color=0xff0000, fmt='(%d,%d)',    fmtArgs=(rectangle.x, rectangle.y))
            ... 		cvui.printf(frame, cvui.mouse().x + 5, cvui.mouse().y + 5, fontScale=0.3, color=0xff0000, fmt='w:%d, h:%d', fmtArgs=(rectangle.width, rectangle.height))
            ...
            ... 	# Did any mouse button go up?
            ... 	if cvui.mouse(query=cvui.UP):
            ... 		# Hide the rectangle
            ... 		rectangle.x = 0
            ... 		rectangle.y = 0
            ... 		rectangle.width = 0
            ... 		rectangle.height = 0
            ...
            ... 	# Was the mouse clicked (any button went down then up)?
            ... 	if cvui.mouse(query=cvui.CLICK):
            ... 		cvui.text(frame, 10, 70, 'Mouse was clicked!')
            ...
            ... 	# Render the rectangle
            ... 	cvui.rect(frame, rectangle.x, rectangle.y, rectangle.width, rectangle.height, 0xff0000)
            ...
            ... 	# This function must be called *AFTER* all UI components. It does
            ... 	# all the behind the scenes magic to handle mouse clicks, etc, then
            ... 	# shows the frame in a window like cv2.imshow() does.
            ... 	cvui.imshow(WINDOW_NAME, frame)
            ...
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    if query is None:
        return __internal.mouseW(windowName or "")
    if button is None:
        if windowName is None:
            return __internal.mouseQ(query)
        else:
            return __internal.mouseWQ(windowName, query)
    else:
        if windowName is None:
            return __internal.mouseBQ(button, query)
        else:
            return __internal.mouseWBQ(windowName, button, query)


def button(
    where=None,
    x=0,
    y=0,
    label="",
    width=0,
    height=30,
    idle=None,
    over=None,
    down=None,
    color=(50, 50, 50),
):
    """Display a button whose graphics are images (np.ndarray). The button accepts three images to describe its states, which are idle (no mouse interaction), over (mouse is over the button) and down (mouse clicked the button). The button size will be defined by the width and height of the images.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            width (int)        : Width of the button.
            height (int)       : Height of the button.
            label (str)        : Text displayed inside the button.
            idle (np.ndarray)  : An image that will be rendered when the button is not interacting with the mouse cursor.
            over (np.ndarray)  : An image that will be rendered when the mouse cursor is over the button.
            down (np.ndarray)  : An image that will be rendered when the mouse cursor clicked the button (or is clicking).

    Returns
            flag (bool) : ``True`` everytime the user clicks the button.

    Examples:
            >>> #=== If you use "width", "height", "label" ===
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= 'Button shortcut'
            >>> frame = np.zeros((150, 650, 3), np.uint8)
            >>> # Init cvui and tell it to use a value of 20 for cv2.waitKey()
            >>> # because we want to enable keyboard shortcut for
            >>> # all components, e.g. button with label "&Quit".
            >>> # If cvui has a value for waitKey, it will call
            >>> # waitKey() automatically for us within cvui.update().
            >>> cvui.init(windowNames=WINDOW_NAME, delayWaitKey=20);
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	cvui.text(where=frame, x=40, y=40, text='To exit this app click the button below or press Q (shortcut for the button below).')
            ...
            ... 	# Exit the application if the quit button was pressed.
            ... 	# It can be pressed because of a mouse click or because
            ... 	# the user pressed the "q" key on the keyboard, which is
            ... 	# marked as a shortcut in the button label ("&Quit").
            ... 	if cvui.button(where=frame, x=300, y=80, label="&Quit"):
            ... 		break
            ...
            ... 	# Since cvui.init() received a param regarding waitKey,
            ... 	# there is no need to call cv.waitKey() anymore. cvui.update()
            ... 	# will do it automatically.
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            >>> cv2.destroyWindow(WINDOW_NAME)
            ...
            >>> #=== If you use "idle", "over", "down" ===
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui, SAMPLE_LENA_IMG, cv2read_mpl
            ...
            >>> WINDOW_NAME	= 'Image button'
            >>> frame = np.zeros(shape=(600, 512, 3), dtype=np.uint8)
            >>> idle = cv2.imread(SAMPLE_LENA_IMG)
            >>> down = np.repeat(cv2.imread(SAMPLE_LENA_IMG, cv2.IMREAD_GRAYSCALE).reshape(*idle.shape[:2], 1), repeats=3, axis=2)
            >>> over = cv2read_mpl(SAMPLE_LENA_IMG)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	# Render an image-based button. You can provide images
            ... 	# to be used to render the button when the mouse cursor is
            ... 	# outside, over or down the button area.
            ... 	if cvui.button(frame, 0, 80, idle=idle, over=over, down=down):
            ... 		print('Image button clicked!')
            ...
            ... 	cvui.text(frame, 150, 30, 'This image behaves as a button')
            ...
            ... 	# Render a regular button.
            ... 	if cvui.button(frame, 0, 80, 'Button'):
            ... 		print('Regular button clicked!')
            ...
            ... 	# This function must be called *AFTER* all UI components. It does
            ... 	# all the behind the scenes magic to handle mouse clicks, etc.
            ... 	cvui.update()
            ...
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ...
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    if width * height > 0:
        return __internal.buttonWH(
            block, x, y, width, height, label, color=color, updateLayout=True
        )
    if all([e is not None for e in [idle, over, down]]):
        return __internal.buttonI(block, x, y, idle, over, down, True)
    else:
        return __internal.button(block, x, y, label, color=color)


def image(where=None, x=0, y=0, image=None):
    """Display an image (np.ndarray).

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            image (np.ndarray) : Image to be rendered in the specified destination.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui, SAMPLE_LENA_IMG
            ...
            >>> WINDOW_NAME	= 'Image'
            >>> frame = np.zeros(shape=(1200, 512, 3), dtype=np.uint8)
            >>> img = cv2.imread(SAMPLE_LENA_IMG)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	cvui.beginRow(frame, x=0, y=50)
            ... 	cvui.image(image=img)
            ... 	cvui.endRow()
            ...
            ... 	cvui.beginRow(frame, x=0, y=650)
            ... 	cvui.image(image=img)
            ... 	cvui.endRow()
            ...
            ... 	cvui.update()
            ...
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    __internal.image(block, x, y, image)


def trackbar(
    where=None,
    x=0,
    y=0,
    width=50,
    value=[],
    min=0.0,
    max=25.0,
    segments=1,
    labelfmt="%.1Lf",
    options=0,
    discreteStep=1,
):
    """Display a trackbar for numeric values that the user can increase/decrease by clicking and/or dragging the marker right or left. This component can use different types of data as its value, so it is imperative provide the right label format, e.g. '%d' for ints, otherwise you might end up with weird errors.

    Args:
            where (np.ndarray)    : image/frame where the component should be rendered.
            x (int)               : Position X where the component should be placed.
            y (int)               : Position Y where the component should be placed.
            width (int)           : Width of the trackbar.
            value ([number])      : Array or list of numbers whose first position, i.e. ``value[0]``, will be used to store the current value of the trackbar. It will be modified when the user interacts with the trackbar. Any numeric type can be used, e.g. int, float, long double, etc.
            min (number)          : Minimum value allowed for the trackbar.
            max (number)          : Maximum value allowed for the trackbar.
            segments (int)        : Number of segments the trackbar will have (default is 1). Segments can be seen as groups of numbers in the scale of the trackbar. For example, 1 segment means a single groups of values (no extra labels along the scale), 2 segments mean the trackbar values will be divided in two groups and a label will be placed at the middle of the scale.
            labelfmt (str)        : Formating string that will be used to render the labels. If you are using a trackbar with integers values, for instance, you can use ``%d`` to render labels.
            options (uint)        : Options to customize the behavior/appearance of the trackbar, expressed as a bitset. Available options are defined as ``cvui.TRACKBAR_`` constants and they can be combined using the bitwise ``|`` operand. Available options are ``cvui.TRACKBAR_HIDE_SEGMENT_LABELS`` (do not render segment labels, but do render min/max labels), ``cvui.TRACKBAR_HIDE_STEP_SCALE`` (do not render the small lines indicating values in the scale), ``cvui.TRACKBAR_DISCRETE`` (changes of the trackbar value are multiples of theDiscreteStep param), ``cvui.TRACKBAR_HIDE_MIN_MAX_LABELS`` (do not render min/max labels), ``cvui.TRACKBAR_HIDE_VALUE_LABEL`` (do not render the current value of the trackbar below the moving marker), and ``cvui.TRACKBAR_HIDE_LABELS`` (do not render labels at all).
            discreteStep (number) : Amount that the trackbar marker will increase/decrease when the marker is dragged right/left (if option ``cvui.TRACKBAR_DISCRETE`` is ON)

    Returns:
            float : The current value.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= 'Trackbar'
            >>> intValue = [30]
            >>> ucharValue = [30]
            >>> charValue = [30]
            >>> floatValue = [12.]
            >>> doubleValue = [45.]
            >>> doubleValue2 = [15.]
            >>> doubleValue3 = [10.3]
            >>> frame = np.zeros((770, 350, 3), np.uint8)
            >>> width = 300
            >>> x = 10
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	# The trackbar component uses templates to guess the type of its arguments.
            ... 	# You have to be very explicit about the type of the value, the min and
            ... 	# the max params. For instance, if they are double, use 100.0 instead of 100.
            ... 	cvui.text(frame, x, 10, 'double, step 1.0 (default)')
            ... 	cvui.trackbar(frame, x, 40, width, doubleValue, 0., 100.)
            ...
            ... 	cvui.text(frame, x, 120, 'float, step 1.0 (default)')
            ... 	cvui.trackbar(frame, x, 150, width, floatValue, 10., 15.)
            ...
            ... 	# You can specify segments and custom labels. Segments are visual marks in
            ... 	# the trackbar scale. Internally the value for the trackbar is stored as
            ... 	# long double, so the custom labels must always format long double numbers, no
            ... 	# matter the type of the numbers being used for the trackbar. E.g. %.2Lf
            ... 	cvui.text(frame, x, 230, 'double, 4 segments, custom label %.2Lf')
            ... 	cvui.trackbar(frame, x, 260, width, doubleValue2, 0., 20., 4, '%.2Lf')
            ...
            ... 	# Again: you have to be very explicit about the value, the min and the max params.
            ... 	# Below is a uchar trackbar. Observe the uchar cast for the min, the max and
            ... 	# the step parameters.
            ... 	cvui.text(frame, x, 340, 'uchar, custom label %.0Lf')
            ... 	cvui.trackbar(frame, x, 370, width, ucharValue, 0, 255, 0, '%.0Lf')
            ...
            ... 	# You can change the behavior of any tracker by using the options parameter.
            ... 	# Options are defined as a bitfield, so you can combine them.
            ... 	# E.g.
            ... 	# TRACKBAR_DISCRETE							# value changes are discrete
            ... 	# TRACKBAR_DISCRETE | TRACKBAR_HIDE_LABELS	# discrete changes and no labels
            ... 	cvui.text(frame, x, 450, 'double, step 0.1, option TRACKBAR_DISCRETE')
            ... 	cvui.trackbar(frame, x, 480, width, doubleValue3, 10., 10.5, 1, '%.1Lf', cvui.TRACKBAR_DISCRETE, 0.1)
            ...
            ... 	# More customizations using options.
            ... 	options = cvui.TRACKBAR_DISCRETE | cvui.TRACKBAR_HIDE_SEGMENT_LABELS
            ... 	cvui.text(frame, x, 560, 'int, 3 segments, DISCRETE | HIDE_SEGMENT_LABELS')
            ... 	cvui.trackbar(frame, x, 590, width, intValue, 10, 50, 3, '%.0Lf', options, 2)
            ...
            ... 	# Trackbar using char type.
            ... 	cvui.text(frame, x, 670, 'char, 2 segments, custom label %.0Lf')
            ... 	cvui.trackbar(frame, x, 700, width, charValue, -128, 127, 2, '%.0Lf')
            ...
            ... 	# This function must be called *AFTER* all UI components. It does
            ... 	# all the behind the scenes magic to handle mouse clicks, etc.
            ... 	cvui.update()
            ...
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ...
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    params = TrackbarParams(min, max, discreteStep, segments, labelfmt, options)
    return __internal.trackbar(block, x, y, width, value, params)


def window(where=None, x=0, y=0, width=640, height=480, title=""):
    """Display a window (a block with a title and a body).

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            width (int)        : Width of the window.
            height (int)       : Height of the window.
            title (str)        : Text displayed as the title of the window.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= 'Matryoshka'
            >>> frame = np.zeros(shape=(150, 150, 3), dtype=np.uint8)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	cvui.beginRow(frame, 10, 10, 100, 100)
            ... 	cvui.text(frame, 0, 5, text="This is Matryoshka.")
            ... 	for i in range(1,5):
            ... 		cvui.window(frame, i*20, i*20, 80-i*20, 80-i*20, f"window{i:>02}")
            ... 	cvui.endRow()
            ...
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    __internal.window(block, x, y, width, height, title)


def rect(
    where=None,
    x=0,
    y=0,
    width=160,
    height=120,
    borderColor=0xFF0000,
    fillingColor=0xFF000000,
):
    """Display a filled rectangle.

    Args:
            where (np.ndarray)  : image/frame where the component should be rendered.
            x (int)             : Position X where the component should be placed.
            y (int)             : Position Y where the component should be placed.
            width (int)         : Width of the rectangle.
            height (int)        : Height of the rectangle.
            borderColor (uint)  : Color of rectangle's border in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
            fillingColor (uint) : Color of rectangle's filling in the format `0xAARRGGBB`, e.g. `0x00ff0000` for red, `0xff000000` for transparent filling.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> def group(frame, x, y, width, height, group_title):
            ... 	padding = 5
            ... 	w = (width - padding) / 4
            ... 	h = (height - 15 - padding) / 2
            ... 	pos = cvui.Point(x + padding, y + 5)
            ...
            ... 	cvui.text(frame, pos.x, pos.y, text=group_title)
            ... 	pos.y += 15
            ...
            ... 	cvui.window(frame, pos.x, pos.y, width - padding * 2, h - padding, "Something")
            ... 	cvui.rect(frame, pos.x + 2, pos.y + 20, width - padding * 2 - 5, h - padding - 20, 0xff0000)
            ... 	pos.y += h
            ...
            ... 	cvui.window(frame, pos.x, pos.y, w / 3 - padding, h, "Some")
            ... 	cvui.text(frame, pos.x + 25, pos.y + 60, text="65", fontScale=1.1)
            ... 	pos.x += w / 3
            ...
            ... 	cvui.window(frame, pos.x, pos.y, w / 3 - padding, h, "Info")
            ... 	cvui.text(frame, pos.x + 25, pos.y + 60, text="30", fontScale=1.1)
            ... 	pos.x += w / 3
            ...
            ... 	cvui.window(frame, pos.x, pos.y, w / 3 - padding, h, "Here")
            ... 	cvui.text(frame, pos.x + 25, pos.y + 60, text="70", fontScale=1.1)
            ... 	pos.x += w / 3
            ...
            ... 	cvui.window(frame, pos.x, pos.y, w - padding, h, "And")
            ... 	cvui.rect(frame, pos.x + 2, pos.y + 22, w - padding - 5, h - padding - 20, 0xff0000)
            ... 	pos.x += w
            ...
            ... 	cvui.window(frame, pos.x, pos.y, w - padding, h, "Here")
            ... 	cvui.rect(frame, pos.x + 2, pos.y + 22, w - padding - 5, h - padding - 20, 0xff0000)
            ... 	pos.x += w
            ...
            ... 	cvui.window(frame, pos.x, pos.y, w - padding, h, "More info")
            ... 	cvui.rect(frame, pos.x + 2, pos.y + 22, w - padding - 5, h - padding - 20, 0xff0000)
            ... 	pos.x += w
            ...
            ... height = 220
            ... spacing = 10
            ... num_groups = 3
            ... frame = np.zeros((height*num_groups, 1300, 3), np.uint8)
            ...
            ... WINDOW_NAME = "Complex layout"
            ... cvui.init(WINDOW_NAME)
            ...
            ... while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	rows, cols, channels = frame.shape
            ...
            ... 	# Render three groups of components.
            ... 	y = 0
            ... 	for i in range(num_groups):
            ... 		group(frame, 0, y, cols, height-spacing, f"Group{i+1:>02}")
            ... 		y += height
            ...
            ... 	# This function must be called *AFTER* all UI components. It does
            ... 	# all the behind the scenes magic to handle mouse clicks, etc.
            ... 	cvui.update()
            ...
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ...
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == 27:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)

    +-----------------------------------------+
    |               Output                    |
    +=========================================+
    | .. image:: _images/opencv.cvui.rect.jpg |
    +-----------------------------------------+
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    __internal.rect(block, x, y, width, height, borderColor, fillingColor)


def sparkline(where=None, x=0, y=0, values=[], width=160, height=120, color=0x00FF00):
    """Display the values of a vector as a sparkline.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            values ([number])  : Array or List containing the numeric values to be used in the sparkline.
            width (int)        : Width of the rectangle.
            height (int)       : Height of the rectangle.
            color (uint)       : Color of sparkline in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= 'Sparkline'
            >>> frame = np.zeros((600, 800, 3), np.uint8)
            >>> cvui.init(WINDOW_NAME)
            >>> rnd = np.random.RandomState()
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	for i,points in enumerate(rnd.uniform(low=0., high=300., size=(3,50))):
            ... 		cvui.sparkline(frame, 100*i, 100*i, points, 800-200*i, 100, 0xff0000 >> i);
            ... 	for j,points in enumerate(rnd.uniform(low=0., high=300., size=(3,500))):
            ... 		cvui.sparkline(frame, 0, 100*(i+j+1), points, 800, 30+30*j, 0xff0000 >> j);
            ...
            ... 	# This function must be called *AFTER* all UI components. It does
            ... 	# all the behind the scenes magic to handle mouse clicks, etc.
            ... 	cvui.update()
            ...
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ...
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)

    +----------------------------------------------+
    |               Output                         |
    +==============================================+
    | .. image:: _images/opencv.cvui.sparkline.gif |
    +----------------------------------------------+
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    __internal.sparkline(block, x, y, values, width, height, color)


def colorpalette(where=None, x=0, y=0, bgr=[], width=300, height=50):
    """Display a color palette using :meth:`trackbar <veditor.opencv.cvui.Internal.trackbar>` .

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            bgr (list)         : Array or list of BGR color.
            width (int)        : Width of the each trackbar and the example image.
            height (int)       : Height of the example image.

    Returns:
            tuple : The current BGR value.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME = 'Color Palette'
            >>> frame = np.zeros((250, 350, 3), np.uint8)
            >>> bgr = [128, 128, 128]
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ... 	# More customizations using options.
            ... 	bgr = cvui.colorpalette(where=frame, x=10, y=10, bgr=bgr, width=300, height=50)
            ... 	cvui.update()
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)

    +-------------------------------------------------+
    |               Output                            |
    +=================================================+
    | .. image:: _images/opencv.cvui.colorpalette.gif |
    +-------------------------------------------------+
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    params = TrackbarParams(
        min=0,
        max=255,
        step=1,
        segments=3,
        labelfmt="%.0Lf",
        options=TRACKBAR_DISCRETE | TRACKBAR_HIDE_SEGMENT_LABELS,
    )
    return __internal.colorpalette(block, x, y, list(bgr), width, height, params)


def drawingpad(
    where=None, x=0, y=0, image=None, color=0xFFFFFF, fillingColor=0x000000, thickness=3
):
    """Create a drawing pad.

    Args:
            where (np.ndarray)  : image/frame where the component should be rendered.
            x (int)             : Position X where the component should be placed.
            y (int)             : Position Y where the component should be placed.
            image (np.ndarray)  : Image to be rendered in the specified destination.
            color (uint)        : Color of the line in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
            fillingColor (uint) : Color of filling in the format `0xAARRGGBB`, e.g. `0x00ff0000` for red, `0xff000000` for transparent filling.
            thickness (int)     : Thickness of the lines used to draw a line.

    Returns:
            np.ndarray : The current ``image`` .

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME = 'Drawing Pad'
            >>> frame = np.zeros(shape=(400, 650, 3), dtype=np.uint8)
            >>> image = np.full(shape=(250,250,3), fill_value=255, dtype=np.uint8)
            >>> bgr = [128, 128, 128]
            >>> fillingColors = ["White", "Black"]
            >>> fillingStates = [True, False]
            >>> thickness = [3]
            >>> cvui.init(WINDOW_NAME)
            >>> cv2.moveWindow(winname=WINDOW_NAME, x=0, y=0)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ... 	cvui.text(where=frame, x=320, y=10,  text="Thickness")
            ... 	cvui.text(where=frame, x=320, y=100, text="Filling Color")
            ... 	thick = cvui.trackbar(where=frame,   x=320, y=30, width=300, value=thickness, min=1, max=10, options=cvui.TRACKBAR_DISCRETE, discreteStep=1)
            ... 	idx = cvui.radiobox(where=frame,     x=350, y=120, labels=fillingColors, states=fillingStates)
            ... 	bgr = cvui.colorpalette(where=frame, x=320, y=180, bgr=bgr, width=300, height=50)
            ... 	image = cvui.drawingpad(where=frame, x=30,  y=50, image=image, color=bgr, fillingColor=[0xffffff, 0x000000][idx], thickness=thick)
            ... 	cvui.update()
            ... 	# Show everything on the screen
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	# Check if ESC key was pressed
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 			break
            >>> cv2.destroyWindow(WINDOW_NAME)

            >>> # You can draw a picture as follows by executing the following program while running the above program.
            >>> def drawing(path, dsize=(250,250), thresh=127, sleep=3, drawing_val=0, offset=(30,125)):
            ...     \"\"\"
            ...     Args:
            ...         path (str)        : Path to binary image.
            ...         dsize (tuple)     : The size of drawing pad. ( ``width`` , ``height`` )
            ...         thresh (int)      : If you prepare the binary (bgr) image, you can use ``cv2.threshold`` to convert it to binary image. (See :meth:`cvPencilSketch <veditor.cli.cvPencilSketch.cvPencilSketch>` for more details.)
            ...         sleep (int)       : Delay execution for a given number of seconds. (You have to click the OpenCV window before before entering the for-loop.)
            ...         drawing_val (int) : At what value to draw.
            ...         offset (tuple)    : Offset from top left ( ``cv2.moveWindow(winname=WINDOW_NAME, x=0, y=0)`` ) to drawing pad.
            ...     \"\"\"
            ... 	import cv2
            ... 	import time
            ... 	import pyautogui as pgui # Use for controling the mouse. (https://pyautogui.readthedocs.io/en/latest/mouse.html)
            ... 	img = cv2.resize(src=cv2.imread(path, 0), dsize=dsize)
            ... 	img = cv2.threshold(src=img, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
            ...     WINDOW_NAME = "Apotheosis"
            ... 	cv2.imshow(winname=v, mat=img)
            ... 	width,height = dsize
            ... 	x_offset, y_offset = offset
            ... 	time.sleep(sleep)
            ... 	for i in range(height):
            ... 		pgui.moveTo(x_offset, y_offset+i)
            ... 		prev_val, prev_pos = (0, 0)
            ... 		for j in range(width+1):
            ... 			if j<width:
            ... 				val = img[i,j]
            ... 			else:
            ... 				val = -1 # Must be different from ``prev_val``
            ... 			if prev_val != val:
            ... 				# Drawing.
            ... 				if prev_val == drawing_val:
            ... 					pgui.mouseDown()
            ... 					pgui.dragRel(xOffset=j-prev_pos, yOffset=0, button="left", duration=0.0, mouseDownUp=True)
            ... 					pgui.mouseUp()
            ... 				else:
            ... 					pgui.moveRel(xOffset=j-prev_pos, yOffset=0, duration=0.0)
            ... 				prev_pos = j
            ... 				prev_val = val
            ... 			key = cv2.waitKey(1)
            ... 			if key == 27: break
            ... 		if key == 27: break
            ...     cv2.destroyWindow(WINDOW_NAME)

    +--------------------------------------------------------+-------------------------------------------------------+
    |                                                     Example                                                    |
    +========================================================+=======================================================+
    | .. image:: _images/opencv.cvui.drawingpad-konotaro.gif | .. image:: _images/opencv.cvui.drawingpad-tanziro.gif |
    +--------------------------------------------------------+-------------------------------------------------------+
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    return __internal.drawingpad(block, x, y, image, color, fillingColor, thickness)


def beginRow(where=None, x=0, y=0, width=-1, height=-1, padding=0, bgColor=None):
    """Start a new row.

    One of the most annoying tasks when building UI is to calculate where each component should be placed on the screen. cvui has a set of methods that abstract the process of positioning components, so you don't have to think about assigning a ``x`` and ``y`` coordinate. Instead you just add components and cvui will place them as you go.

    You use ``beginRow()`` to start a group of elements. After ``beginRow()`` has been called, all subsequent component calls don't have to specify the frame where the component should be rendered nor its position. The position of the component will be automatically calculated by cvui based on the components within the group. All components are placed side by side, from left to right

    NOTE: Don't forget to call ``endRow()`` to finish the row, otherwise cvui will throw an error.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            width (int)        : Width of the row. If a negative value is specified, the width of the row will be automatically calculated based on the content of the block.
            height (int)       : Height of the row. If a negative value is specified, the height of the row will be automatically calculated based on the content of the block.
            padding (int)      : Space, in pixels, among the components of the block.
            bgColor (tuple)    : Background Color.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= 'Nested columns'
            >>> frame = np.zeros((600, 800, 3), np.uint8)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	# Define a row at position (10, 50) with width 100 and height 150.
            ... 	cvui.beginRow(frame, 10, 50, width=100, height=150)
            ... 	# The components below will be placed one beside the other.
            ... 	cvui.text(text='Row starts')
            ... 	cvui.button(label='btn')
            ... 	cvui.text(text='This is the end of the row!')
            ... 	cvui.endRow()
            ...
            ... 	# Here is another nested row/column
            ... 	cvui.beginRow(frame, 50, 300, 100, 150)
            ... 	cvui.button(label='btn2')
            ... 	cvui.text(text='This is second row.')
            ... 	if cvui.button(label='&Quit'):
            ... 		break
            ... 	cvui.endRow()
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    __internal.begin(ROW, block.where, x, y, width, height, padding, bgColor)


def endRow():
    """End a row. You must call this function only if you have previously called its counter part, the ``beginRow()`` function."""
    __internal.end(ROW)


def beginColumn(where=None, x=0, y=0, width=-1, height=-1, padding=0, bgColor=None):
    """Start a new column.

    One of the most annoying tasks when building UI is to calculate where each component should be placed on the screen. cvui has a set of methods that abstract the process of positioning components, so you don't have to think about assigning a X and Y coordinate. Instead you just add components and cvui will place them as you go.

    You use ``beginColumn()`` to start a group of elements. After ``beginColumn()`` has been called, all subsequent component calls don't have to specify the frame where the component should be rendered nor its position. The position of the component will be automatically calculated by cvui based on the components within the group. All components are placed below each other, from the top of the screen towards the bottom.

    Args:
            where (np.ndarray) : image/frame where the component should be rendered.
            x (int)            : Position X where the component should be placed.
            y (int)            : Position Y where the component should be placed.
            width (int)        : Width of the column. If a negative value is specified, the width of the column will be automatically calculated based on the content of the block.
            height (int)       : Height of the column. If a negative value is specified, the height of the column will be automatically calculated based on the content of the block.
            padding (int)      : Space, in pixels, among the components of the block.
            bgColor (tuple)    : Background Color.

    Examples:
            >>> import cv2
            >>> import numpy as np
            >>> from vedirot.cli import cvui
            ...
            >>> WINDOW_NAME	= 'Nested columns'
            >>> frame = np.zeros((600, 800, 3), np.uint8)
            >>> cvui.init(WINDOW_NAME)
            ...
            >>> while (True):
            ... 	# Fill the frame with a nice color
            ... 	frame[:] = (49, 52, 49)
            ...
            ... 	# Define a row at position (10, 50) with width 100 and height 150.
            ... 	cvui.beginRow(frame, 10, 50, 100, 150)
            ...
            ... 	cvui.beginColumn(width=100, height=150)
            ... 	cvui.text(text='Column starts')
            ... 	cvui.button(label='btn')
            ... 	cvui.text(text='This is the end of the column!')
            ... 	cvui.endColumn()
            ...
            ... 	# Here is another nested row/column
            ... 	cvui.beginColumn(width=100, height=150)
            ... 	cvui.button(label='btn2')
            ... 	cvui.text(text='This is second column.')
            ... 	if cvui.button(label='&Quit'):
            ... 		break
            ... 	cvui.endColumn()
            ...
            ... 	cvui.endRow()
            ...
            ... 	cvui.update()
            ... 	cv2.imshow(WINDOW_NAME, frame)
            ...
            ... 	if cv2.waitKey(20) == cvui.ESCAPE:
            ... 		break
            >>> cv2.destroyWindow(WINDOW_NAME)
    """
    handleTypeError(types=[np.ndarray, NoneType], where=where)
    if isinstance(where, np.ndarray):
        __internal.screen.where = where
        block = __internal.screen
    else:
        block = __internal.topBlock()
        x += block.anchor.x
        y += block.anchor.y

    __internal.begin(COLUMN, block.where, x, y, width, height, padding, bgColor)


def endColumn():
    """End a column. You must call this function only if you have previously called its counter part, i.e. ``beginColumn()``"""
    __internal.end(COLUMN)


class Point:
    """Represent a 2D point.

    Attributes:
            x (int)      : Position X.
            y (int)      : Position Y.
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def inside(self, rect):
        return rect.contains(self)


class Rect:
    """Represent a rectangle.

    Attributes:
            x (int)      : Position X.
            y (int)      : Position Y.
            width (int)  : Width.
            height (int) : Height.
    """

    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, thePoint):
        return (self.x <= thePoint.x <= (self.x + self.width)) and (
            self.y <= thePoint.y <= (self.y + self.height)
        )

    def area(self):
        return self.width * self.height


class Size(Rect):
    """Represent the size of something, i.e. width and height.

    It is essentially a simplified version of :py:class: `Rect <veditor.opencv.cvui.Rect>` where x and y are zero.
    """

    def __init__(self, width=0, height=0):
        self.x = 0
        self.y = 0
        self.width = width
        self.height = height


class Block:
    """Describe a block structure used by cvui to handle ``begin*()`` and ``end*()`` calls.

    Attributes:
            where (np.ndarray) : Where the block should be rendered to.
            rect (Rect)        : The size and position of the block.
            fill (Rect)        : The filled area occuppied by the block as it gets modified by its inner components.
            anchor (Point)     : The point where the next component of the block should be rendered.
            padding (int)      : Padding among components within this block.
            type (int)         : Type of the block, e.g. ``cvui.ROW`` or ``cvui.COLUMN``.
    """

    def __init__(self):
        self.where = None
        self.rect = Rect()
        self.fill = Rect()
        self.anchor = Point()
        self.padding = 0
        self.type = ROW
        self.reset()

    def reset(self):
        self.rect.x = 0
        self.rect.y = 0
        self.rect.width = 0
        self.rect.height = 0

        self.fill = self.rect
        self.fill.width = 0
        self.fill.height = 0

        self.anchor.x = 0
        self.anchor.y = 0

        self.padding = 0


class Label:
    """Describe a component label, including info about a shortcut.

    If a label contains ``"Re&start"``

    Attributes:
            hasShortcut (bool)       : ``label[i]==&`` and ``i!=-1``
            shortcut (str)           : character after ``"&"`` ( ``label[i+1]`` )
            textBeforeShortcut (str) : Text before shortcut ( ``label[:i]`` )
            textAfterShortcut (str ) : Text after shortcut ( ``label[i+1:]`` )
    """

    def __init__(self):
        self.hasShortcut = False
        self.shortcut = ""
        self.textBeforeShortcut = ""
        self.textAfterShortcut = ""


class MouseButton:
    """Describe a mouse button

    Attributes:
            justReleased (bool) : Whether the mouse button was released, i.e. click event.
            justPressed (bool)  : Whether the mouse button was just pressed, i.e. true for a frame when a button is down.
            pressed (bool)      : Whether the mouse button is pressed or not.
    """

    def __init__(self):
        self.justReleased = False
        self.justPressed = False
        self.pressed = False

    def reset(self):
        self.justPressed = False
        self.justReleased = False
        self.pressed = False


class Mouse:
    """Describe the information of the mouse cursor

    Attributes:
            buttons (dict)          : Status of each button. Use ``cvui.{RIGHT,LEFT,MIDDLE}_BUTTON`` to access the buttons.
            anyButton (MouseButton) : Represent the behavior of all mouse buttons combined
            position (Point)        : x and y coordinates of the mouse at the moment.
    """

    def __init__(self):
        self.buttons = {
            LEFT_BUTTON: MouseButton(),
            MIDDLE_BUTTON: MouseButton(),
            RIGHT_BUTTON: MouseButton(),
        }
        self.anyButton = MouseButton()
        self.position = Point(0, 0)


class Context:
    """Describe a (window) context.

    Attributes:
            windowName (str) : Name of the window related to this context.
            mouse (Mouse)    : The mouse cursor related to this context.
    """

    def __init__(self):
        self.windowName = ""
        self.mouse = Mouse()


class TrackbarParams:
    """Describe the inner parts of the trackbar component.

    Attributes:
            min (number)   : Minimum value allowed for the trackbar.
            max (number)   : Maximum value allowed for the trackbar.
            step (number)  : Amount that should be increased/decreased when the user interacts with the counter buttons
            segments (int) : Number of segments the trackbar will have (default is 1). Segments can be seen as groups of numbers in the scale of the trackbar. For example, 1 segment means a single groups of values (no extra labels along the scale), 2 segments mean the trackbar values will be divided in two groups and a label will be placed at the middle of the scale.
            labelfmt (str) : Formating string that will be used to render the labels. If you are using a trackbar with integers values, for instance, you can use ``%d`` to render labels.
            options (uint) : Options to customize the behavior/appearance of the trackbar, expressed as a bitset. Available options are defined as ``cvui.TRACKBAR_`` constants and they can be combined using the bitwise ``|`` operand. Available options are ``cvui.TRACKBAR_HIDE_SEGMENT_LABELS`` (do not render segment labels, but do render min/max labels), ``cvui.TRACKBAR_HIDE_STEP_SCALE`` (do not render the small lines indicating values in the scale), ``cvui.TRACKBAR_DISCRETE`` (changes of the trackbar value are multiples of theDiscreteStep param), ``cvui.TRACKBAR_HIDE_MIN_MAX_LABELS`` (do not render min/max labels), ``cvui.TRACKBAR_HIDE_VALUE_LABEL`` (do not render the current value of the trackbar below the moving marker), and ``cvui.TRACKBAR_HIDE_LABELS`` (do not render labels at all).
    """

    def __init__(
        self, min=0.0, max=255.0, step=1.0, segments=0, labelfmt="%.0Lf", options=0
    ):
        self.min = min
        self.max = max
        self.step = step
        self.segments = segments
        self.options = options
        self.labelfmt = labelfmt


class Internal:
    """This class contains all stuff that cvui uses internally to render and control interaction with components.

    Attributes:
            defaultContext (str)      : Default window name.
            currentContext (str)      : Current (active) window name.
            contexts (dict)           : Indexed by the window name.
            lastKeyPressed (int)      : Last key that was pressed. TODO: collect it per window
            lastMousePosition (Point) : Last mouse position when that was pressed. TODO: collect it per window
            delayWaitKey (int)        : Delay value (milliseconds) passed to ``cv2.waitKey()``. If a negative value is informed (default is ``-1``), cvui will not automatically call ``cv2.waitKey()`` within ``cvui.update()``, which will disable keyboard shortcuts for all components. If you want to enable keyboard shortcut for components (e.g. using & in a button label), you must specify a positive value for this param.
            screen (Block)            : Block structure.
            stack (list)              : Block stack.
            trackbarMarginX (int)     : X-axis Margin of trackbar.
            _render (Render)          : contains all rendering methods. ( ``_render._internal = self`` )
    """

    def __init__(self):
        self.defaultContext = ""
        self.currentContext = ""
        self.contexts = {}
        # self.buffer = []
        self.lastKeyPressed = -1
        self.lastMousePosition = Point(-1, -1)
        self.delayWaitKey = -1
        self.screen = Block()
        self.stack = []
        self.trackbarMarginX = 14

        self._render = Render()
        self._render._internal = self

    def isMouseButton(self, button, query):
        """Get to know the relationship between the mouse and the ``button`` by returning the ``button`` state according to the ``query``"""
        ret = False

        if query == CLICK or query == UP:
            ret = button.justReleased
        elif query == DOWN:
            ret = button.justPressed
        elif query == IS_DOWN:
            ret = button.pressed

        return ret

    def mouseW(self, windowName=""):
        """Return the last position of the mouse.

        Args:
                windowName (str) : Name of the window whose mouse cursor will be used. If nothing is informed (default), the function will return the position of the mouse cursor for the default window (the one informed in ``cvui.init()`` ).

        Returns:
                point (Point) : A point containing the position of the mouse cursor in the speficied window.
        """
        return self.getContext(windowName).mouse.position

    def mouseQ(self, query):
        """
        Query the mouse for events, e.g. "is any button down now?". Available queries are:

        - ``cvui.DOWN``: any mouse button was pressed. ``cvui.mouse`` returns ``True`` for a single frame only.
        - ``cvui.UP``: any mouse button was released.  ``cvui.mouse`` returns ``True`` for a single frame only.
        - ``cvui.CLICK``: any mouse button was clicked (went down then up, no matter the amount of frames in between). ``cvui.mouse`` returns ``True`` for a single frame only.
        - ``cvui.IS_DOWN``: any mouse button is currently pressed. ``cvui.mouse`` returns ``True`` for as long as the button is down/pressed.

        It is easier to think of this function as the answer to a questions. For instance, asking if any mouse button went down:

        .. code-block:: python

                >>> if (cvui.mouse(cvui.DOWN)):
                ... 	# Any mouse button just went down.

        The window whose mouse will be queried depends on the context. If ``cvui.mouse(query)`` is being called after ``cvui.context()``, the window informed in the context will be queried. If no context is available, the default window (informed in ``cvui.init()`` ) will be used.

        Args:
                query (int) : An integer describing the intended mouse query. Available queries are ``cvui.DOWN``, ``cvui.UP``, ``cvui.CLICK``, and ``cvui.IS_DOWN``.

        Returns:
                isMouseButton (bool) : Whether the ``button`` and ``query`` are match.
        """
        return self.mouseWQ("", query)

    def mouseWQ(self, windowName, query):
        """Query the mouse for events in a particular window. This function behave exactly like ``cvui.mouse(query)`` with the difference that queries are targeted at a particular window.

        Args:
                windowName (str) : Name of the window that will be queried.
                query (int)      : An integer describing the intended mouse query. Available queries are ``cvui.DOWN``, ``cvui.UP``, ``cvui.CLICK``, and ``cvui.IS_DOWN``.

        Returns:
                isMouseButton (bool) : Whether the ``button`` and ``query`` are match.
        """
        button = self.getContext(windowName).mouse.anyButton
        return self.isMouseButton(button, query)

    def mouseBQ(self, button, query):
        """Query the mouse for events in a particular button. This function behave exactly like ``cvui.mouse(int query)``, with the difference that queries are targeted at a particular mouse button instead.

        Args:
                button (int) : An integer describing the mouse button to be queried. Possible values are ``cvui.LEFT_BUTTON``, ``cvui.MIDDLE_BUTTON`` and ``cvui.LEFT_BUTTON``.
                query  (int) : An integer describing the intended mouse query. Available queries are ``cvui.DOWN``, ``cvui.UP``, ``cvui.CLICK``, and ``cvui.IS_DOWN``.

        Returns:
                isMouseButton (bool) : Whether the ``button`` and ``query`` are match.
        """
        return self.mouseWBQ("", button, query)

    def mouseWBQ(self, windowName, button, query):
        """Query the mouse for events in a particular button in a particular window. This function behave exactly like ``cvui.mouse(int button, int query)``, with the difference that queries are targeted at a particular mouse button in a particular window instead.

        Args:
                windowName (str) : Name of the window that will be queried.
                button (int)     : An integer describing the mouse button to be queried. Possible values are ``cvui.LEFT_BUTTON``, ``cvui.MIDDLE_BUTTON`` and ``cvui.LEFT_BUTTON``.
                query (int)      : An integer describing the intended mouse query. Available queries are ``cvui.DOWN``, ``cvui.UP``, ``cvui.CLICK``, and ``cvui.IS_DOWN``.

        Returns:
                isMouseButton (bool) : Whether the ``button`` and ``query`` are match.
        """
        if button != RIGHT_BUTTON and button != MIDDLE_BUTTON and button != LEFT_BUTTON:
            __internal.error(
                6,
                "Invalid mouse button. Are you using one of the available: cvui.{RIGHT,MIDDLE,LEFT}_BUTTON ?",
            )

        button = self.getContext(windowName).mouse.buttons[button]
        return self.isMouseButton(button, query)

    def init(self, windowName, delayWaitKey):
        """Initiallize the window.

        Args:
                windowNames (str,list)    : Array containing the name of the windows where components will be added. Those windows will be automatically if `createNamedWindows` is `True`.
                delayWaitKey (int)        : Delay value passed to ``cv2.waitKey()``. If a negative value is informed (default is ``-1``), cvui will not automatically call ``cv2.waitKey()`` within ``cvui.update()``, which will disable keyboard shortcuts for all components. If you want to enable keyboard shortcut for components (e.g. using & in a button label), you must specify a positive value for this param.
        """
        self.defaultContext = windowName
        self.currentContext = windowName
        self.delayWaitKey = delayWaitKey
        self.lastKeyPressed = -1

    def bitsetHas(self, bitset, value):
        """Check whether ``bitset`` has the signal of ``value`` e.g. ``cvui.TRACKBAR_HIDE_VALUE_LABEL``

        Args:
                bitset (int) : Binary value with various signals (= ``1`` )
                value (int)  : Binary value with only one ``1``

        Returns:
                bool : Whether ``bitset`` has the signal of ``value`` or not.
        """
        return (bitset & value) != 0

    def error(self, errorId, message):
        """Output the Error message.

        Args:
                errorId (int) : Error Id used in cvui.
                message (str) : An additional message.
        """
        print(toRED(f"[CVUI] Fatal error (code {errorId}) :"), message)
        cv2.waitKey(100000)
        sys.exit(-1)

    def getContext(self, windowName=""):
        """Get Context.

        Args:
                windowName (str) : Name of the window whose UI interactions will be tracked.
        """
        if len(windowName) != 0:
            # Get context in particular
            return self.contexts[windowName]

        elif len(self.currentContext) != 0:
            # No window provided, return currently active context.
            return self.contexts[self.currentContext]

        elif len(self.defaultContext) != 0:
            # We have no active context, so let's use the default one.
            return self.contexts[self.defaultContext]

        else:
            # Apparently we have no window at all! <o>
            # This should not happen. Probably cvui::init() was never called.
            self.error(5, "Unable to read context. Did you forget to call cvui.init()?")

    def updateLayoutFlow(self, block, size):
        """Update layot for additional Rows or Columns.

        Args:
                block (Block) : A block structure.
                size (Size)   : Represent the size of something.
        """
        if block.type == ROW:
            aValue = size.width + block.padding

            block.anchor.x += aValue
            block.fill.width += aValue
            block.fill.height = max(size.height, block.fill.height)

        elif block.type == COLUMN:
            aValue = size.height + block.padding

            block.anchor.y += aValue
            block.fill.height += aValue
            block.fill.width = max(size.width, block.fill.width)

    @property
    def isblockStackEmpty(self):
        """Is block stack ( ``self.stack`` ) is empty or not."""
        return len(self.stack) == 0

    @property
    def stackCount(self):
        """The numeber of blocks in stack ( ``self.stack`` )"""
        return len(self.stack)

    def topBlock(self):
        """Get the top (last) ``Block`` in ``self.stack``"""
        if self.isblockStackEmpty:
            self.error(
                3,
                "You are using a function that should be enclosed by begin*() and end*(), but you probably forgot to call begin*().",
            )
        return self.stack[-1]

    def pushBlock(self, block):
        """Push a ``Block`` object into ``self.stack``

        Args:
                block (Block) : A block structure.
        """
        self.stack.append(block)

    def popBlock(self):
        """Pop a ``Block`` object from ``self.stack``"""
        # Check if there is anything to be popped out from the stack.
        if self.isblockStackEmpty:
            self.error(
                1,
                "Mismatch in the number of begin*()/end*() calls. You are calling one more than the other.",
            )
        return self.stack.pop()

    def createLabel(self, label):
        """Create a Label object.

        Args:
                label (str) : Label string.

        Returns:
                Label : Describe a component label, including info about a shortcut.
        """
        aLabel = Label()
        i = label[:-1].find("&")
        if i == -1:
            aLabel.hasShortcut = False
            aLabel.shortcut = ""
            aLabel.textBeforeShortcut = label
            aLabel.textAfterShortcut = ""
        else:
            aLabel.hasShortcut = True
            aLabel.shortcut = label[i + 1]
            aLabel.textBeforeShortcut = label[:i]
            aLabel.textAfterShortcut = label[i + 2 :]
        return aLabel

    def text(
        self,
        block,
        x,
        y,
        text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=0xCECECE,
        thickness=1,
        lineType=cv2.LINE_8,
        updateLayout=True,
    ):
        """Display a piece of text.

        Args:
                block (Block)       : A block structure.
                x (int)             : Position X where the component should be placed.
                y (int)             : Position Y where the component should be placed.
                text (str)          : The text content.
                fontFace (int)      : Font type. (default= ``cv2.FONT_HERSHEY_SIMPLEX`` )
                fontScale (float)   : Font scale factor that is multiplied by the font-specific base size.
                color (uint)        : Color of the text in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
                thickness (int)     : Thickness of the lines used to draw a text.
                lineType (int)      : Line type. (default= ``cv2.LINE_8`` )
                updateLayout (bool) : Whether updates layot or not.
        """
        (text_width, text_height), _ = cv2.getTextSize(
            text=text, fontFace=fontFace, fontScale=fontScale, thickness=thickness
        )

        text_size = Size(text_width, text_height)
        aPos = Point(x, y + text_size.height)

        self._render.text(
            block,
            text,
            aPos,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=lineType,
        )

        if updateLayout:
            # Add an extra pixel to the height to overcome OpenCV font size problems.
            text_size.height += 1
            self.updateLayoutFlow(block, text_size)

    def counter(self, block, x, y, value, step, fmt):
        """Display a counter for integer values that the user can increase/descrease by clicking the up and down arrows.

        Args:
                block (Block)    : A block structure.
                x (int)          : Position X where the component should be placed.
                y (int)          : Position Y where the component should be placed.
                value ([number]) : Array or List of numbers whose first position, i.e. ``value[0]``, will be used to store the current value of the counter.
                step (number)    : Amount that should be increased/decreased when the user interacts with the counter buttons
                fmt (str)        : How the value of the counter should be presented, as it was printed by ``stdio's printf()``. E.g. ``'%d'`` means the value will be displayed as an integer, ``'%0d'`` integer with one leading zero, etc.

        Returns:
                value (number) : Number that corresponds to the current value of the counter.
        """
        aContentArea = Rect(x + 22, y, 48, 22)

        if self.buttonWH(
            block=block,
            x=x,
            y=y,
            width=22,
            height=22,
            label="-",
            color=(209, 198, 138),
            updateLayout=False,
        ):
            value[0] -= step

        aText = fmt % value[0]
        self._render.counter(block, aContentArea, aText)

        if self.buttonWH(
            block=block,
            x=aContentArea.x + aContentArea.width,
            y=y,
            width=22,
            height=22,
            label="+",
            color=(185, 182, 255),
            updateLayout=False,
        ):
            value[0] += step

        # Update the layout flow
        size = Size(22 * 2 + aContentArea.width, aContentArea.height)
        self.updateLayoutFlow(block, size)

        return value[0]

    def checkbox(self, block, x, y, label, state, color):
        mouse = self.getContext().mouse
        aRect = Rect(x, y, 15, 15)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        text_size = Rect(0, 0, text_width, text_height)
        aHitArea = Rect(x, y, aRect.width + text_size.width + 6, aRect.height)
        mouseIsOver = aHitArea.contains(mouse.position)

        if mouseIsOver:
            self._render.checkbox(block=block, state=OVER, shape=aRect)
            if mouse.anyButton.justReleased:
                state[0] = not state[0]
        else:
            self._render.checkbox(block=block, state=OUT, shape=aRect)

        self._render.checkboxLabel(block, aRect, label, text_size, color)

        if state[0]:
            self._render.checkboxCheck(block=block, shape=aRect)

        # Update the layout flow
        size = Size(aHitArea.width, aHitArea.height)
        self.updateLayoutFlow(block, size)

        return state[0]

    def radiobox(self, block, x, y, labels, states, color):
        if len(labels) != len(states):
            self.error(
                6,
                f"`states` and `labels` should have the same length, but got ({len(labels)}!={len(states)})",
            )
        if states.count(True) != 1:
            self.error(
                6,
                f"`states` should have only 1 True value, but got ({states.count(True)})",
            )

        mouse = self.getContext().mouse

        crt_state = -1
        width = 0
        for i, label in enumerate(labels):
            aRect = Rect(x, y + i * 20, 15, 15)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            text_size = Rect(0, 0, text_width, text_height)
            aHitArea = Rect(
                x, y + i * 20, aRect.width + text_size.width + 6, aRect.height
            )
            if crt_state == -1 and aHitArea.contains(mouse.position):
                self._render.checkbox(block, OVER, aRect)
                if mouse.anyButton.justReleased:
                    # if mouse.anyButton.pressed:
                    crt_state = i
                    states[crt_state] = True
            else:
                self._render.checkbox(block, OUT, aRect)

            self._render.checkboxLabel(block, aRect, label, text_size, color)
            width = max(aHitArea.width, width)
        height = (aHitArea.y - y) + aHitArea.height

        # Choose only one of a predefined set of mutually exclusive options.
        if crt_state >= 0:
            for i in range(len(states)):
                if i != crt_state:
                    states[i] = False

        for i, state in enumerate(states):
            if state:
                aRect = Rect(x, y + i * 20, 15, 15)
                self._render.checkboxCheck(block, aRect)

        # Update the layout flow
        self.updateLayoutFlow(block, Size(width, height))

        return states.index(True)

    def clip(self, value, v_min=0.0, v_max=1.0):
        """Given an interval, values outside the interval are clipped to the interval edges.

        Args:
                value (number) : value.
                v_min (number) : Minimum value.
                v_max (number) : Maximum value.

        Returns:
                number : Clipped value.
        """
        return min(max(value, v_min), v_max)

    def trackbar(self, block, x, y, width, value, params):
        """Display a trackbar for numeric values that the user can increase/decrease by clicking and/or dragging the marker right or left. This component can use different types of data as its value, so it is imperative provide the right label format, e.g. '%d' for ints, otherwise you might end up with weird errors.

        Args:
                block (Block)           : A block structure.
                x (int)                 : Position X where the component should be placed.
                y (int)                 : Position Y where the component should be placed.
                width (int)             : Width of the trackbar.
                value ([number])        : Array or list of numbers whose first position, i.e. ``value[0]``, will be used to store the current value of the trackbar. It will be modified when the user interacts with the trackbar. Any numeric type can be used, e.g. int, float, long double, etc.
                params (TrackbarParams) : Describe the inner parts of the trackbar component.

        Returns:
                float : The current value.
        """
        mouse = self.getContext().mouse
        aContentArea = Rect(x, y, width, 45)
        mouseIsOver = aContentArea.contains(mouse.position)
        aValue = value[0]

        state = OVER if mouseIsOver else OUT, aContentArea
        self._render.trackbar(
            block=block, state=state, shape=aContentArea, value=value[0], params=params
        )

        if mouse.anyButton.pressed and mouseIsOver:
            value[0] = self.trackbarXPixelToValue(
                params=params, bounding=aContentArea, pixelX=mouse.position.x
            )

            if self.bitsetHas(params.options, TRACKBAR_DISCRETE):
                self.trackbarForceValuesAsMultiplesOfSmallStep(params, value)

        # Update the layout flow
        # TODO: use size = aContentArea.size()?
        self.updateLayoutFlow(block, aContentArea)

        return value[0]

    def trackbarForceValuesAsMultiplesOfSmallStep(self, params, value):
        """Correct the value when the trackbar allows only discrete changes ( ``cvui.TRACKBAR_DISCRETE`` )

        Args:
                params (TrackbarParams) : Describe the inner parts of the trackbar component.
                value ([number])        : Array or List of numbers whose first position, i.e. ``value[0]``, will be used to store the current value of the counter.
        """
        if self.bitsetHas(params.options, TRACKBAR_DISCRETE) and params.step != 0.0:
            k = float(value[0] - params.min) / params.step
            k = round(k)
            value[0] = params.min + params.step * k

    def trackbarXPixelToValue(self, params, bounding, pixelX):
        """Calculate the value from the position of the trackbar.

        Args:
                params (TrackbarParams) : Describe the inner parts of the trackbar component.
                bounding (Rect)         : The area of trackbar.
                pixelX (int)            : The position in the trackbar.

        Returns:
                float : The current value.
        """
        aRatio = float(pixelX - (bounding.x + self.trackbarMarginX)) / (
            bounding.width - 2 * self.trackbarMarginX
        )
        aRatio = self.clip(aRatio, v_min=0.0, v_max=1.0)
        aValue = params.min + aRatio * (params.max - params.min)
        return aValue

    def trackbarValueToXPixel(self, params, bounding, value):
        """Calculate the position in the trackbar from the current value.

        Args:
                params (TrackbarParams) : Describe the inner parts of the trackbar component.
                bounding (Rect)         : The area of trackbar.
                value (float)           : The current value in the trackbar.

        Returns:
                int : The position in the trackbar.
        """
        aRatio = float(value - params.min) / (params.max - params.min)
        aRatio = self.clip(aRatio, v_min=0.0, v_max=1.0)
        aPixelsX = (
            bounding.x
            + self.trackbarMarginX
            + aRatio * (bounding.width - 2 * self.trackbarMarginX)
        )
        return int(aPixelsX)

    def colorpalette(self, block, x, y, bgr, width, height, params):
        """Display a color palette using :meth:`trackbar <veditor.opencv.cvui.Internal.trackbar>` .

        Args:
                block (Block)  : A block structure.
                x (int)        : Position X where the component should be placed.
                y (int)        : Position Y where the component should be placed.
                bgr (list)     : Array or list of BGR color.
                width (int)    : Width of the each trackbar and the example image.
                height (int)   : Height of the example image.
                params (TrackbarParams) : Describe the inner parts of the trackbar component.

        Returns:
                tuple : The current BGR value.
        """
        for i, (val, label) in enumerate(zip(bgr, "BGR")):
            color = [0, 0, 0]
            color[i] = val
            self.rect(
                block=block,
                x=x,
                y=y + 10,
                width=20,
                height=20,
                borderColor=0xCCCCCC,
                fillingColor=0xFFFFFF,
            )
            self.text(
                block=block,
                x=x + 5,
                y=y + 15,
                text=label,
                color=color,
                fontScale=1,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
            )
            bgr[i] = self.trackbar(
                block=block, x=x + 15, y=y, width=width, value=[val], params=params
            )
            y += 50
        self.image(
            block=block,
            x=x + 15,
            y=y + 10,
            image=np.full(shape=(height, width, 3), fill_value=bgr, dtype=np.uint8),
        )
        self.text(
            block=block,
            x=x + 20,
            y=y + 15,
            text="Sample",
            color=choose_text_color(color=color, max_val=255, is_bgr=True),
        )
        return bgr

    def iarea(self, x, y, width, height):
        """Create an interaction area that reports activity with the mouse cursor. The tracked interactions are returned by the function and they are:

        Args:
                x (int)      : Position X where the interactive area should be placed.
                y (int)      : Position Y where the interactive area should be placed.
                width (int)  : Width of the interactive area.
                height (int) : Height of the interactive area.

        Returns:
                int : An integer value representing the current state of interaction with the mouse cursor. It can be
                                                - ``cvui.OUT`` when the cursor is not over the iarea.
                                                - ``cvui.OVER`` when the cursor is over the iarea.
                                                - ``cvui.DOWN`` when the cursor is pressed over the iarea, but not released yet.
                                                - ``cvui.CLICK`` when the cursor clicked (pressed and released) within the iarea.
        """
        mouse = self.getContext().mouse
        # By default, return that the mouse is out of the interaction area.
        ret = OUT
        # Check if the mouse is over the interaction area.
        mouseIsOver = Rect(x, y, width, height).contains(mouse.position)
        if mouseIsOver:
            if mouse.anyButton.pressed:
                ret = DOWN
            else:
                ret = OVER
        # Tell if the button was clicked or not
        if mouseIsOver and mouse.anyButton.justReleased:
            # if mouseIsOver and mouse.anyButton.pressed:
            ret = CLICK
        return ret

    def buttonWH(
        self, block, x, y, width, height, label, color=(50, 50, 50), updateLayout=True
    ):
        """Create a bottun using ``width`` and ``height`` .

        Args:
                block (Block)       : A block structure.
                x (int)             : Position X where the component should be placed.
                y (int)             : Position Y where the component should be placed.
                width (int)         : Width of the button.
                height (int)        : Height of the button.
                label (str)         : Text displayed inside the button.
                color (tuple)       : Button color.
                updateLayout (bool) : Whether updates layot or not.

        Returns:
                bool: Whether the button was clicked or not.
        """
        # Calculate the space that the label will fill
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        text_size = Rect(0, 0, text_width, text_height)

        # Make the button big enough to house the label
        aRect = Rect(x, y, width, height)

        # Render the button according to mouse interaction, e.g. OVER, DOWN, OUT.
        aStatus = self.iarea(x, y, aRect.width, aRect.height)
        button_bgr = self.hex2bgr(color)
        self._render.button(block, aStatus, aRect, label, color=button_bgr)
        self._render.buttonLabel(
            block,
            aStatus,
            aRect,
            label,
            text_size,
            color=choose_text_color(color=button_bgr, max_val=255, is_bgr=True),
        )

        # Update the layout flow according to button size
        # if we were told to update.
        if updateLayout:
            size = Size(width, height)
            self.updateLayoutFlow(block, size)

        # Handle keyboard shortcuts
        aWasShortcutPressed = False
        if self.lastKeyPressed != -1:
            aLabel = self.createLabel(label)
            if (
                aLabel.hasShortcut
                and aLabel.shortcut.lower() == chr(self.lastKeyPressed).lower()
            ):
                aWasShortcutPressed = True
        return aStatus == CLICK or aWasShortcutPressed

    def button(self, block, x, y, label, color=(50, 50, 50)):
        """Create a bottun using :meth:`buttonWH <veditor.opencv.cvui.buttonWH>` by calculating ``width`` and ``height`` from ``label``.

        Args:
                block (Block)       : A block structure.
                x (int)             : Position X where the component should be placed.
                y (int)             : Position Y where the component should be placed.
                label (str)         : Text displayed inside the button.
                color (tuple)       : Button color.

        Returns:
                bool: Whether the button was clicked or not.
        """
        # Calculate the space that the label will fill
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        text_size = Rect(0, 0, text_width, text_height)

        # Create a button based on the size of the text
        return self.buttonWH(
            block,
            x,
            y,
            width=text_size.width + 30,
            height=text_size.height + 18,
            label=label,
            color=color,
            updateLayout=True,
        )

    def buttonI(self, block, x, y, idle, over, down, updateLayout=True):
        """Create a bottun using images.

        Args:
                block (Block)     : A block structure.
                x (int)           : Position X where the component should be placed.
                y (int)           : Position Y where the component should be placed.
                idle (np.ndarray) : An image that will be rendered when the button is not interacting with the mouse cursor.
                over (np.ndarray) : An image that will be rendered when the mouse cursor is over the button.
                down (np.ndarray) : An image that will be rendered when the mouse cursor clicked the button (or is clicking).

        Returns:
                bool: Whether the button was clicked or not.
        """
        idle_height, idle_width = idle.shape[:2]
        aRect = Rect(x, y, idle_width, idle_height)
        aStatus = self.iarea(x, y, aRect.width, aRect.height)

        if aStatus == OUT:
            self._render.image(block, aRect, idle)
        elif aStatus == OVER:
            self._render.image(block, aRect, over)
        elif aStatus == DOWN:
            self._render.image(block, aRect, down)

        # Update the layout flow according to button size
        # if we were told to update.
        if updateLayout:
            size = Size(aRect.width, aRect.height)
            self.updateLayoutFlow(block, size)

        return aStatus == CLICK

    def image(self, block, x, y, image):
        """Display an image (np.ndarray).

        Args:
                block (Block)      : A block structure.
                x (int)            : Position X where the component should be placed.
                y (int)            : Position Y where the component should be placed.
                image (np.ndarray) : Image to be rendered in the specified destination.
        """
        img_height, img_width = image.shape[:2]
        aRect = Rect(x, y, img_width, img_height)

        # TODO: check for render outside the frame area
        self._render.image(block, aRect, image)

        # Update the layout flow according to image size
        size = Size(img_width, img_height)
        self.updateLayoutFlow(block, size)

    def window(self, block, x, y, width, height, title):
        """Display a window (a block with a title and a body).

        Args:
                block (Block) : A block structure.
                x (int)       : Position X where the component should be placed.
                y (int)       : Position Y where the component should be placed.
                width (int)   : Width of the window.
                height (int)  : Height of the window.
                title (str)   : Text displayed as the title of the window.
        """
        aTitleBar = Rect(x, y, width, 20)
        aContent = Rect(x, y + aTitleBar.height, width, height - aTitleBar.height)

        self._render.window(block, aTitleBar, aContent, title)

        # Update the layout flow
        size = Size(width, height)
        self.updateLayoutFlow(block, size)

    def rect(self, block, x, y, width, height, borderColor, fillingColor):
        """Display a filled rectangle.

        Args:
                block (Block)       : A block structure.
                x (int)             : Position X where the component should be placed.
                y (int)             : Position Y where the component should be placed.
                width (int)         : Width of the rectangle.
                height (int)        : Height of the rectangle.
                borderColor (uint)  : Color of rectangle's border in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
                fillingColor (uint) : Color of rectangle's filling in the format `0xAARRGGBB`, e.g. `0x00ff0000` for red, `0xff000000` for transparent filling.
        """
        aAnchor = Point(x, y)
        aRect = Rect(x, y, width, height)

        aRect.x = aAnchor.x + aRect.width if aRect.width < 0 else aAnchor.x
        aRect.y = aAnchor.y + aRect.height if aRect.height < 0 else aAnchor.y
        aRect.width = abs(aRect.width)
        aRect.height = abs(aRect.height)

        self._render.rect(block, aRect, borderColor, fillingColor)

        # Update the layout flow
        size = Size(aRect.width, aRect.height)
        self.updateLayoutFlow(block, size)

    def drawingpad(
        self, block, x, y, image, color=0xFFFFFF, fillingColor=0x000000, thickness=3
    ):
        """Create a drawing pad.

        Args:
                block (Block)       : A block structure.
                x (int)             : Position X where the component should be placed.
                y (int)             : Position Y where the component should be placed.
                image (np.ndarray)  : Image to be rendered in the specified destination.
                color (uint)        : Color of the line in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
                thickness (int)     : Thickness of the lines used to draw a line.
                fillingColor (uint) : Color of filling in the format `0xAARRGGBB`, e.g. `0x00ff0000` for red, `0xff000000` for transparent filling.

        Returns:
                np.ndarray : The current ``image`` .
        """
        mouse = self.getContext().mouse
        img_height, img_width = image.shape[:2]
        aHitArea = Rect(x, y, img_width, img_height)
        if aHitArea.contains(mouse.position) and mouse.anyButton.pressed:
            lx, ly = self.lastMousePosition.x, self.lastMousePosition.y
            if lx != -1:
                cv2.line(
                    image,
                    pt1=(lx - x, ly - y),
                    pt2=(mouse.position.x - x, mouse.position.y - y),
                    color=self.hex2bgr(color),
                    thickness=thickness,
                )
            self.lastMousePosition.x, self.lastMousePosition.y = (
                mouse.position.x,
                mouse.position.y,
            )
        else:
            self.lastMousePosition = Point(-1, -1)

        fillingColor = self.hex2bgr(fillingColor)[:3]
        if self.buttonWH(
            block=block,
            x=x + img_width // 2 - 50,
            y=y + img_height + 10,
            width=100,
            height=22,
            label="Fill",
            color=fillingColor,
            updateLayout=False,
        ):
            image[:] = fillingColor

        self.image(block=block, x=x, y=y, image=image)
        return image

    def sparkline(self, block, x, y, values, width, height, color):
        """Display the values of a vector as a sparkline.

        Args:
                block (Block)      : A block structure.
                x (int)            : Position X where the component should be placed.
                y (int)            : Position Y where the component should be placed.
                values ([number])  : Array or List containing the numeric values to be used in the sparkline.
                width (int)        : Width of the rectangle.
                height (int)       : Height of the rectangle.
                color (uint)       : Color of sparkline in the format ``0xRRGGBB``, e.g. ``0xff0000`` for red.
        """
        aRect = Rect(x, y, width, height)
        aHowManyValues = len(values)

        if aHowManyValues >= 2:
            aMin, aMax = self.findMinMax(values)
            self._render.sparkline(block, values, aRect, aMin, aMax, color)
        else:
            self.text(
                block,
                x,
                y,
                "No data." if aHowManyValues == 0 else "Insufficient data points.",
                fontScale=0.4,
                color=0xCECECE,
                updateLayout=False,
            )

        # Update the layout flow
        size = Size(width, height)
        self.updateLayoutFlow(block, size)

    def hex2bgr(self, color):
        """Convert hex ( ``int`` ) color to ``tuple`` color.

        Args:
                color (int/hex) : Color.

        Returns:
                tuple : ``tuple`` formatted color.
        """
        if isinstance(color, int):
            aAlpha = (color >> 24) & 0xFF
            aRed = (color >> 16) & 0xFF
            aGreen = (color >> 8) & 0xFF
            aBlue = color & 0xFF
            color = (aBlue, aGreen, aRed, aAlpha)
        return color

    def begin(self, type, where, x, y, width, height, padding, bgColor=None):
        """Begin a row or column."""
        if bgColor is not None:
            where[y : y + height, x : x + width, :] = bgColor

        block = Block()
        block.where = where

        block.rect.x = x
        block.rect.y = y
        block.rect.width = width
        block.rect.height = height

        block.fill = block.rect
        block.fill.width = 0
        block.fill.height = 0

        block.anchor.x = x
        block.anchor.y = y

        block.padding = padding
        block.type = type
        self.pushBlock(block)

    def end(self, type):
        """End a row or column."""
        block = self.popBlock()

        if block.type != type:
            self.error(
                4,
                "Calling wrong type of end*(). E.g. endColumn() instead of endRow(). Check if your begin*() calls are matched with their appropriate end*() calls.",
            )

        # If we still have blocks in the stack, we must update
        # the current top with the dimensions that were filled by
        # the newly popped block.

        if not self.isblockStackEmpty:
            top_block = self.topBlock()
            size = Size()

            # If the block has rect.width < 0 or rect.heigth < 0, it means the
            # user don't want to calculate the block's width/height. It's up to
            # us do to the math. In that case, we use the block's fill rect to find
            # out the occupied space. If the block's width/height is greater than
            # zero, then the user is very specific about the desired size. In that
            # case, we use the provided width/height, no matter what the fill rect
            # actually is.
            size.width = block.fill.width if block.rect.width < 0 else block.rect.width
            size.height = (
                block.fill.height if block.rect.height < 0 else block.rect.height
            )

            self.updateLayoutFlow(top_block, size)

    def findMinMax(self, values):
        """Find the min and max values of a vector.

        Args:
                values (list) : Array shaped vectors.

        Returns:
                tuple : the min and max values of a vector
        """
        aMin = values[0]
        aMax = values[0]

        for aValue in values:
            if aValue < aMin:
                aMin = aValue

            if aValue > aMax:
                aMax = aValue

        return (aMin, aMax)


class Render:
    """Class that contains all rendering methods."""

    _internal = None

    def rectangle(self, where, shape, color, thickness=1, LineType=CVUI_ANTIALISED):
        aStartPoint = (int(shape.x), int(shape.y))
        aEndPoint = (int(shape.x + shape.width), int(shape.y + shape.height))

        cv2.rectangle(where, aStartPoint, aEndPoint, color, thickness, LineType)

    def text(
        self,
        block,
        text,
        position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=0xCECECE,
        thickness=1,
        lineType=cv2.LINE_8,
    ):
        aPosition = (int(position.x), int(position.y))
        cv2.putText(
            img=block.where,
            text=text,
            org=aPosition,
            fontFace=fontFace,
            fontScale=fontScale,
            color=self._internal.hex2bgr(color),
            thickness=thickness,
            lineType=lineType,
        )

    def counter(
        self,
        block,
        shape,
        value,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=0xCECECE,
        thickness=1,
        lineType=cv2.LINE_AA,
    ):
        self.rectangle(block.where, shape, (0x29, 0x29, 0x29), CVUI_FILLED)  # fill
        self.rectangle(block.where, shape, (0x45, 0x45, 0x45))  # border

        (text_width, text_height), _ = cv2.getTextSize(
            text=value, fontFace=fontFace, fontScale=fontScale, thickness=thickness
        )
        text_size = Rect(0, 0, text_width, text_height)

        aPos = Point(
            shape.x + shape.width / 2 - text_size.width / 2,
            shape.y + text_size.height / 2 + shape.height / 2,
        )
        cv2.putText(
            block.where,
            value,
            (int(aPos.x), int(aPos.y)),
            fontFace=fontFace,
            fontScale=fontScale,
            color=self._internal.hex2bgr(color),
            thickness=thickness,
            lineType=lineType,
        )

    def button(self, block, state, shape, label, color=(50, 50, 50)):
        # Outline
        self.rectangle(block.where, shape, (0x29, 0x29, 0x29))

        # Border
        shape.x += 1
        shape.y += 1
        shape.width -= 2
        shape.height -= 2
        self.rectangle(block.where, shape, (0x4A, 0x4A, 0x4A))

        # Inside
        shape.x += 1
        shape.y += 1
        shape.width -= 2
        shape.height -= 2

        colors = generate_color_series(
            color=self._internal.hex2bgr(color), variation=3, diff=35
        )
        if state == OUT:
            inside_color = colors[1]
        elif state == OVER:
            inside_color = colors[2]
        else:
            inside_color = colors[0]
        self.rectangle(
            where=block.where, shape=shape, color=inside_color, thickness=CVUI_FILLED
        )

    def image(self, block, rect, image):
        if image.ndim == 2 and block.where.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        block.where[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] = image

    def putText(self, block, state, color, text, position):
        text_width = 0
        if text != "":
            afontScale = 0.39 if state == DOWN else 0.4
            aPosition = (int(position.x), int(position.y))
            cv2.putText(
                block.where,
                text,
                aPosition,
                cv2.FONT_HERSHEY_SIMPLEX,
                afontScale,
                color,
                1,
                CVUI_ANTIALISED,
            )
            (text_width, _), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, afontScale, 1
            )
        return text_width

    def putTextCentered(self, block, position, text):
        afontScale = 0.3

        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, afontScale, 1
        )
        text_size = Rect(0, 0, text_width, text_height)
        aPositionDecentered = Point(position.x - text_size.width / 2, position.y)
        cv2.putText(
            block.where,
            text,
            (int(aPositionDecentered.x), int(aPositionDecentered.y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            afontScale,
            (0xCE, 0xCE, 0xCE),
            1,
            CVUI_ANTIALISED,
        )

        return text_size.width

    def buttonLabel(
        self, block, state, rect, label, textSize, color=(0xCE, 0xCE, 0xCE)
    ):
        aPos = Point(
            rect.x + rect.width / 2 - textSize.width / 2,
            rect.y + rect.height / 2 + textSize.height / 2,
        )

        aLabel = self._internal.createLabel(label)

        if aLabel.hasShortcut == False:
            self.putText(block, state, color, label, aPos)

        else:
            aWidth = self.putText(block, state, color, aLabel.textBeforeShortcut, aPos)
            aStart = aPos.x + aWidth
            aPos.x += aWidth

            aShortcut = ""
            aShortcut += aLabel.shortcut

            aWidth = self.putText(block, state, color, aShortcut, aPos)
            aEnd = aStart + aWidth
            aPos.x += aWidth

            self.putText(block, state, color, aLabel.textAfterShortcut, aPos)
            cv2.line(
                block.where,
                (int(aStart), int(aPos.y + 3)),
                (int(aEnd), int(aPos.y + 3)),
                color,
                1,
                CVUI_ANTIALISED,
            )

    def trackbarHandle(self, block, state, shape, value, params, workingArea):
        aBarTopLeft = Point(workingArea.x, workingArea.y + workingArea.height / 2)
        aBarHeight = 7

        # Draw the rectangle representing the handle
        aPixelX = self._internal.trackbarValueToXPixel(params, shape, value)
        aIndicatorWidth = 3
        aIndicatorHeight = 4
        aPoint1 = Point(aPixelX - aIndicatorWidth, aBarTopLeft.y - aIndicatorHeight)
        aPoint2 = Point(
            aPixelX + aIndicatorWidth, aBarTopLeft.y + aBarHeight + aIndicatorHeight
        )
        aRect = Rect(aPoint1.x, aPoint1.y, aPoint2.x - aPoint1.x, aPoint2.y - aPoint1.y)

        aFillColor = 0x525252 if state == OVER else 0x424242

        self.rect(block, aRect, 0x212121, 0x212121)
        aRect.x += 1
        aRect.y += 1
        aRect.width -= 2
        aRect.height -= 2
        self.rect(block, aRect, 0x515151, aFillColor)

        aShowLabel = (
            self._internal.bitsetHas(params.options, TRACKBAR_HIDE_VALUE_LABEL) == False
        )

        # Draw the handle label
        if aShowLabel:
            aTextPos = Point(aPixelX, aPoint2.y + 11)
            aText = params.labelfmt % value
            self.putTextCentered(block, aTextPos, aText)

    def trackbarPath(self, block, state, shape, value, params, workingArea):
        aBarHeight = 7
        aBarTopLeft = Point(workingArea.x, workingArea.y + workingArea.height / 2)
        aRect = Rect(aBarTopLeft.x, aBarTopLeft.y, workingArea.width, aBarHeight)

        aBorderColor = 0x4E4E4E if state == OVER else 0x3E3E3E

        self.rect(block, aRect, aBorderColor, 0x292929)
        cv2.line(
            block.where,
            (int(aRect.x + 1), int(aRect.y + aBarHeight - 2)),
            (int(aRect.x + aRect.width - 2), int(aRect.y + aBarHeight - 2)),
            (0x0E, 0x0E, 0x0E),
        )

    def trackbarSteps(self, block, state, shape, value, params, workingArea):
        aBarTopLeft = Point(workingArea.x, workingArea.y + workingArea.height / 2)
        aColor = (0x51, 0x51, 0x51)

        aDiscrete = self._internal.bitsetHas(params.options, TRACKBAR_DISCRETE)
        aFixedStep = params.step if aDiscrete else (params.max - params.min) / 20

        # TODO: check min, max and step to prevent infinite loop.
        aValue = params.min
        while aValue <= params.max:
            aPixelX = int(self._internal.trackbarValueToXPixel(params, shape, aValue))
            aPoint1 = (aPixelX, int(aBarTopLeft.y))
            aPoint2 = (aPixelX, int(aBarTopLeft.y - 3))
            cv2.line(block.where, aPoint1, aPoint2, aColor)
            aValue += aFixedStep

    def trackbarSegmentLabel(self, block, shape, params, value, workingArea, showLabel):
        aColor = (0x51, 0x51, 0x51)
        aBarTopLeft = Point(workingArea.x, workingArea.y + workingArea.height / 2)

        aPixelX = int(self._internal.trackbarValueToXPixel(params, shape, value))

        aPoint1 = (aPixelX, int(aBarTopLeft.y))
        aPoint2 = (aPixelX, int(aBarTopLeft.y - 8))
        cv2.line(block.where, aPoint1, aPoint2, aColor)

        if showLabel:
            aText = params.labelfmt % value
            aTextPos = Point(aPixelX, aBarTopLeft.y - 11)
            self.putTextCentered(block, aTextPos, aText)

    def trackbarSegments(self, block, state, shape, value, params, workingArea):
        aSegments = 1 if params.segments < 1 else params.segments
        aSegmentLength = float(params.max - params.min) / aSegments

        aHasMinMaxLabels = (
            self._internal.bitsetHas(params.options, TRACKBAR_HIDE_MIN_MAX_LABELS)
            == False
        )

        # Render the min value label
        self.trackbarSegmentLabel(
            block, shape, params, params.min, workingArea, aHasMinMaxLabels
        )

        # Draw large steps and labels
        aHasSegmentLabels = (
            self._internal.bitsetHas(params.options, TRACKBAR_HIDE_SEGMENT_LABELS)
            == False
        )
        # TODO: check min, max and step to prevent infinite loop.
        aValue = params.min
        while aValue <= params.max:
            self.trackbarSegmentLabel(
                block, shape, params, aValue, workingArea, aHasSegmentLabels
            )
            aValue += aSegmentLength

        # Render the max value label
        self.trackbarSegmentLabel(
            block, shape, params, params.max, workingArea, aHasMinMaxLabels
        )

    def trackbar(self, block, state, shape, value, params):
        aWorkingArea = Rect(
            shape.x + self._internal.trackbarMarginX,
            shape.y,
            shape.width - 2 * self._internal.trackbarMarginX,
            shape.height,
        )

        self.trackbarPath(block, state, shape, value, params, aWorkingArea)

        aHideAllLabels = self._internal.bitsetHas(params.options, TRACKBAR_HIDE_LABELS)
        aShowSteps = (
            self._internal.bitsetHas(params.options, TRACKBAR_HIDE_STEP_SCALE) == False
        )

        if aShowSteps and aHideAllLabels == False:
            self.trackbarSteps(block, state, shape, value, params, aWorkingArea)

        if aHideAllLabels == False:
            self.trackbarSegments(block, state, shape, value, params, aWorkingArea)

        self.trackbarHandle(block, state, shape, value, params, aWorkingArea)

    def checkbox(self, block, state, shape):
        colors = (
            [(0x64, 0x64, 0x64), (0x17, 0x17, 0x17)]
            if state == OUT
            else [(0xC8, 0xC8, 0xC8), (0xFF, 0xFF, 0x00)]
        )

        # Outline
        self.rectangle(block.where, shape, colors[0])
        # Border
        shape.x += 1
        shape.y += 1
        shape.width -= 2
        shape.height -= 2
        self.rectangle(block.where, shape, colors[1])

        # Inside
        shape.x += 1
        shape.y += 1
        shape.width -= 2
        shape.height -= 2
        self.rectangle(block.where, shape, (0x29, 0x29, 0x29), CVUI_FILLED)

    def checkboxLabel(self, block, rect, label, textSize, color):
        aPos = Point(
            rect.x + rect.width + 6,
            rect.y + textSize.height + rect.height / 2 - textSize.height / 2 - 1,
        )
        self.text(block, label, aPos, fontScale=0.4, color=color)

    def checkboxCheck(self, block, shape):
        shape.x += 1
        shape.y += 1
        shape.width -= 2
        shape.height -= 2
        self.rectangle(block.where, shape, (0xFF, 0xBF, 0x75), CVUI_FILLED)

    def window(self, block, titleBar, content, title):
        aTransparecy = False
        aAlpha = 0.3
        aOverlay = block.where.copy()

        # Render borders in the title bar
        self.rectangle(block.where, titleBar, (0x4A, 0x4A, 0x4A))

        # Render the inside of the title bar
        titleBar.x += 1
        titleBar.y += 1
        titleBar.width -= 2
        titleBar.height -= 2
        self.rectangle(block.where, titleBar, (0x21, 0x21, 0x21), CVUI_FILLED)

        # Render title text.
        aPos = Point(titleBar.x + 5, titleBar.y + 12)
        cv2.putText(
            block.where,
            title,
            (int(aPos.x), int(aPos.y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0xCE, 0xCE, 0xCE),
            1,
            CVUI_ANTIALISED,
        )

        # Render borders of the body
        self.rectangle(block.where, content, (0x4A, 0x4A, 0x4A))

        # Render the body filling.
        content.x += 1
        content.y += 1
        content.width -= 2
        content.height -= 2
        self.rectangle(aOverlay, content, (0x31, 0x31, 0x31), CVUI_FILLED)

        if aTransparecy:
            np.copyto(aOverlay, block.where)  # block.where.copyTo(aOverlay);
            self.rectangle(aOverlay, content, (0x31, 0x31, 0x31), CVUI_FILLED)
            cv2.addWeighted(
                aOverlay, aAlpha, block.where, 1.0 - aAlpha, 0.0, block.where
            )
        else:
            self.rectangle(block.where, content, (0x31, 0x31, 0x31), CVUI_FILLED)

    def rect(self, block, position, borderColor, fillingColor):
        aBorderColor = self._internal.hex2bgr(borderColor)
        aFillingColor = self._internal.hex2bgr(fillingColor)

        aHasFilling = aFillingColor[3] != 0xFF

        if aHasFilling:
            self.rectangle(
                block.where, position, aFillingColor, CVUI_FILLED, CVUI_ANTIALISED
            )

        # Render the border
        self.rectangle(block.where, position, aBorderColor)

    def sparkline(self, block, values, rect, min, max, color):
        size = len(values)
        i = 0
        aScale = max - min
        aGap = float(rect.width) / size
        aPosX = rect.x

        while i <= size - 2:
            x = aPosX
            y = (
                (values[i] - min) / aScale * -(rect.height - 5)
                + rect.y
                + rect.height
                - 5
            )
            aPoint1 = Point(x, y)

            x = aPosX + aGap
            y = (
                (values[i + 1] - min) / aScale * -(rect.height - 5)
                + rect.y
                + rect.height
                - 5
            )
            aPoint2 = Point(x, y)

            cv2.line(
                block.where,
                (int(aPoint1.x), int(aPoint1.y)),
                (int(aPoint2.x), int(aPoint2.y)),
                self._internal.hex2bgr(color),
            )
            aPosX += aGap

            i += 1


# Access points to internal global namespaces.
__internal = Internal()
