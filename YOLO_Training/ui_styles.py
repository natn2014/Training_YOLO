"""
UI Styles and Theme Configuration
Provides light and dark theme support with proper contrast for readability
"""

class Theme:
    """Base theme class"""
    def __init__(self, name):
        self.name = name
        
    def get_main_stylesheet(self):
        """Get main window stylesheet"""
        raise NotImplementedError
        
    def get_header_stylesheet(self):
        """Get header stylesheet"""
        raise NotImplementedError
        
    def get_tab_stylesheet(self):
        """Get tab widget stylesheet"""
        raise NotImplementedError
        
    def get_button_stylesheet(self, button_type='default'):
        """Get button stylesheet"""
        raise NotImplementedError
        
    def get_groupbox_stylesheet(self):
        """Get group box stylesheet"""
        raise NotImplementedError
        
    def get_input_stylesheet(self):
        """Get input widget stylesheet"""
        raise NotImplementedError


class LightTheme(Theme):
    """Light/White theme with good contrast"""
    
    def __init__(self):
        super().__init__("Light")
        
        # Color palette
        self.bg_primary = "#FFFFFF"
        self.bg_secondary = "#F5F5F5"
        self.bg_tertiary = "#E8E8E8"
        self.text_primary = "#000000"
        self.text_secondary = "#333333"
        self.accent = "#2196F3"
        self.accent_hover = "#1976D2"
        self.success = "#4CAF50"
        self.success_hover = "#45A049"
        self.danger = "#F44336"
        self.danger_hover = "#D32F2F"
        self.border = "#CCCCCC"
        self.border_light = "#E0E0E0"
        
    def get_main_stylesheet(self):
        return f"""
            QMainWindow {{
                background-color: {self.bg_secondary};
            }}
            QWidget {{
                background-color: {self.bg_primary};
                color: {self.text_primary};
            }}
            QLabel {{
                color: {self.text_primary};
                background-color: transparent;
            }}
        """
        
    def get_header_stylesheet(self):
        return f"""
            font-size: 24px;
            font-weight: bold;
            color: {self.text_primary};
            padding: 15px;
            background-color: {self.bg_tertiary};
            border: 2px solid {self.border};
            border-radius: 5px;
        """
        
    def get_tab_stylesheet(self):
        return f"""
            QTabWidget::pane {{
                border: 2px solid {self.border};
                border-radius: 5px;
                background-color: {self.bg_primary};
            }}
            QTabBar::tab {{
                background: {self.bg_tertiary};
                color: {self.text_primary};
                padding: 10px 20px;
                margin: 2px;
                border-radius: 3px;
                border: 1px solid {self.border_light};
            }}
            QTabBar::tab:selected {{
                background: {self.accent};
                color: #FFFFFF;
                font-weight: bold;
                border: 1px solid {self.accent};
            }}
            QTabBar::tab:hover {{
                background: {self.accent_hover};
                color: #FFFFFF;
            }}
        """
        
    def get_button_stylesheet(self, button_type='default'):
        if button_type == 'success':
            return f"""
                QPushButton {{
                    background-color: {self.success};
                    color: #FFFFFF;
                    font-weight: bold;
                    padding: 10px;
                    border: none;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.success_hover};
                }}
                QPushButton:pressed {{
                    background-color: #3D8B40;
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_tertiary};
                    color: #999999;
                }}
            """
        elif button_type == 'danger':
            return f"""
                QPushButton {{
                    background-color: {self.danger};
                    color: #FFFFFF;
                    font-weight: bold;
                    padding: 10px;
                    border: none;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.danger_hover};
                }}
                QPushButton:pressed {{
                    background-color: #B71C1C;
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_tertiary};
                    color: #999999;
                }}
            """
        elif button_type == 'primary':
            return f"""
                QPushButton {{
                    background-color: {self.accent};
                    color: #FFFFFF;
                    font-weight: bold;
                    padding: 8px 16px;
                    border: none;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.accent_hover};
                }}
                QPushButton:pressed {{
                    background-color: #1565C0;
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_tertiary};
                    color: #999999;
                }}
            """
        else:  # default
            return f"""
                QPushButton {{
                    background-color: {self.bg_tertiary};
                    color: {self.text_primary};
                    padding: 8px 16px;
                    border: 1px solid {self.border};
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.border_light};
                    border: 1px solid {self.accent};
                }}
                QPushButton:pressed {{
                    background-color: {self.border};
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_secondary};
                    color: #999999;
                    border: 1px solid {self.border_light};
                }}
            """
            
    def get_groupbox_stylesheet(self):
        return f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.border};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {self.bg_primary};
                color: {self.text_primary};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.text_primary};
                background-color: {self.bg_primary};
            }}
        """
        
    def get_input_stylesheet(self):
        return f"""
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
                background-color: {self.bg_primary};
                color: {self.text_primary};
                border: 1px solid {self.border};
                border-radius: 3px;
                padding: 5px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {{
                border: 2px solid {self.accent};
            }}
            QLineEdit:read-only {{
                background-color: {self.bg_secondary};
                color: {self.text_secondary};
            }}
            QComboBox::drop-down {{
                border: none;
                background-color: {self.bg_tertiary};
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {self.text_primary};
                margin-right: 5px;
            }}
        """
        
    def get_table_stylesheet(self):
        return f"""
            QTableWidget {{
                background-color: {self.bg_primary};
                color: {self.text_primary};
                gridline-color: {self.border_light};
                border: 1px solid {self.border};
            }}
            QTableWidget::item {{
                padding: 5px;
            }}
            QTableWidget::item:selected {{
                background-color: {self.accent};
                color: #FFFFFF;
            }}
            QHeaderView::section {{
                background-color: {self.bg_tertiary};
                color: {self.text_primary};
                padding: 5px;
                border: 1px solid {self.border};
                font-weight: bold;
            }}
        """
        
    def get_progressbar_stylesheet(self):
        return f"""
            QProgressBar {{
                border: 2px solid {self.border};
                border-radius: 5px;
                text-align: center;
                background-color: {self.bg_secondary};
                color: {self.text_primary};
            }}
            QProgressBar::chunk {{
                background-color: {self.accent};
                border-radius: 3px;
            }}
        """
        
    def get_status_label_stylesheet(self, status='info'):
        if status == 'success':
            return f"font-weight: bold; color: {self.success}; background-color: transparent;"
        elif status == 'error':
            return f"font-weight: bold; color: {self.danger}; background-color: transparent;"
        elif status == 'warning':
            return "font-weight: bold; color: #FF9800; background-color: transparent;"
        else:  # info
            return f"font-weight: bold; color: {self.accent}; background-color: transparent;"


class DarkTheme(Theme):
    """Dark/Black theme with good contrast"""
    
    def __init__(self):
        super().__init__("Dark")
        
        # Color palette
        self.bg_primary = "#1E1E1E"
        self.bg_secondary = "#252525"
        self.bg_tertiary = "#2D2D2D"
        self.text_primary = "#E0E0E0"
        self.text_secondary = "#B0B0B0"
        self.accent = "#42A5F5"
        self.accent_hover = "#64B5F6"
        self.success = "#66BB6A"
        self.success_hover = "#81C784"
        self.danger = "#EF5350"
        self.danger_hover = "#E57373"
        self.border = "#404040"
        self.border_light = "#505050"
        
    def get_main_stylesheet(self):
        return f"""
            QMainWindow {{
                background-color: {self.bg_secondary};
            }}
            QWidget {{
                background-color: {self.bg_primary};
                color: {self.text_primary};
            }}
            QLabel {{
                color: {self.text_primary};
                background-color: transparent;
            }}
        """
        
    def get_header_stylesheet(self):
        return f"""
            font-size: 24px;
            font-weight: bold;
            color: {self.text_primary};
            padding: 15px;
            background-color: {self.bg_tertiary};
            border: 2px solid {self.border};
            border-radius: 5px;
        """
        
    def get_tab_stylesheet(self):
        return f"""
            QTabWidget::pane {{
                border: 2px solid {self.border};
                border-radius: 5px;
                background-color: {self.bg_primary};
            }}
            QTabBar::tab {{
                background: {self.bg_tertiary};
                color: {self.text_primary};
                padding: 10px 20px;
                margin: 2px;
                border-radius: 3px;
                border: 1px solid {self.border_light};
            }}
            QTabBar::tab:selected {{
                background: {self.accent};
                color: #000000;
                font-weight: bold;
                border: 1px solid {self.accent};
            }}
            QTabBar::tab:hover {{
                background: {self.accent_hover};
                color: #000000;
            }}
        """
        
    def get_button_stylesheet(self, button_type='default'):
        if button_type == 'success':
            return f"""
                QPushButton {{
                    background-color: {self.success};
                    color: #000000;
                    font-weight: bold;
                    padding: 10px;
                    border: none;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.success_hover};
                }}
                QPushButton:pressed {{
                    background-color: #A5D6A7;
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_tertiary};
                    color: #606060;
                }}
            """
        elif button_type == 'danger':
            return f"""
                QPushButton {{
                    background-color: {self.danger};
                    color: #000000;
                    font-weight: bold;
                    padding: 10px;
                    border: none;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.danger_hover};
                }}
                QPushButton:pressed {{
                    background-color: #EF9A9A;
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_tertiary};
                    color: #606060;
                }}
            """
        elif button_type == 'primary':
            return f"""
                QPushButton {{
                    background-color: {self.accent};
                    color: #000000;
                    font-weight: bold;
                    padding: 8px 16px;
                    border: none;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.accent_hover};
                }}
                QPushButton:pressed {{
                    background-color: #90CAF9;
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_tertiary};
                    color: #606060;
                }}
            """
        else:  # default
            return f"""
                QPushButton {{
                    background-color: {self.bg_tertiary};
                    color: {self.text_primary};
                    padding: 8px 16px;
                    border: 1px solid {self.border};
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.border_light};
                    border: 1px solid {self.accent};
                }}
                QPushButton:pressed {{
                    background-color: {self.border};
                }}
                QPushButton:disabled {{
                    background-color: {self.bg_secondary};
                    color: #606060;
                    border: 1px solid {self.border_light};
                }}
            """
            
    def get_groupbox_stylesheet(self):
        return f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.border};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {self.bg_primary};
                color: {self.text_primary};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.text_primary};
                background-color: {self.bg_primary};
            }}
        """
        
    def get_input_stylesheet(self):
        return f"""
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
                background-color: {self.bg_secondary};
                color: {self.text_primary};
                border: 1px solid {self.border};
                border-radius: 3px;
                padding: 5px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {{
                border: 2px solid {self.accent};
            }}
            QLineEdit:read-only {{
                background-color: {self.bg_tertiary};
                color: {self.text_secondary};
            }}
            QComboBox::drop-down {{
                border: none;
                background-color: {self.bg_tertiary};
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {self.text_primary};
                margin-right: 5px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                background-color: {self.bg_tertiary};
                border: 1px solid {self.border};
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {self.border_light};
            }}
        """
        
    def get_table_stylesheet(self):
        return f"""
            QTableWidget {{
                background-color: {self.bg_primary};
                color: {self.text_primary};
                gridline-color: {self.border_light};
                border: 1px solid {self.border};
            }}
            QTableWidget::item {{
                padding: 5px;
            }}
            QTableWidget::item:selected {{
                background-color: {self.accent};
                color: #000000;
            }}
            QHeaderView::section {{
                background-color: {self.bg_tertiary};
                color: {self.text_primary};
                padding: 5px;
                border: 1px solid {self.border};
                font-weight: bold;
            }}
        """
        
    def get_progressbar_stylesheet(self):
        return f"""
            QProgressBar {{
                border: 2px solid {self.border};
                border-radius: 5px;
                text-align: center;
                background-color: {self.bg_secondary};
                color: {self.text_primary};
            }}
            QProgressBar::chunk {{
                background-color: {self.accent};
                border-radius: 3px;
            }}
        """
        
    def get_status_label_stylesheet(self, status='info'):
        if status == 'success':
            return f"font-weight: bold; color: {self.success}; background-color: transparent;"
        elif status == 'error':
            return f"font-weight: bold; color: {self.danger}; background-color: transparent;"
        elif status == 'warning':
            return "font-weight: bold; color: #FFA726; background-color: transparent;"
        else:  # info
            return f"font-weight: bold; color: {self.accent}; background-color: transparent;"


class ThemeManager:
    """Manages theme switching and application"""
    
    def __init__(self):
        self.themes = {
            'Light': LightTheme(),
            'Dark': DarkTheme()
        }
        self.current_theme = self.themes['Light']
        
    def set_theme(self, theme_name):
        """Set current theme by name"""
        if theme_name in self.themes:
            self.current_theme = self.themes[theme_name]
            return True
        return False
        
    def get_theme(self):
        """Get current theme"""
        return self.current_theme
        
    def get_available_themes(self):
        """Get list of available theme names"""
        return list(self.themes.keys())
