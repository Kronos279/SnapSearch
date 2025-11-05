import sys
import os
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QStackedWidget, QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox, QMenu, QStyle, QListView
)
from PySide6.QtGui import QPixmap, QIcon, QAction
from PySide6.QtCore import (
    Slot, QSize, Qt, QRunnable, QThreadPool, QObject, Signal, QPoint
)

import folderManager
import build_index
import main as pipeline_main
import preprocessing_images

APP_STYLESHEET = """
    QWidget {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    QMainWindow, QWidget#MainWidget {
        background-color: #ffffff;
    }

    QWidget#Sidebar {
        background-color: #f8f8f8;
        border-right: 1px solid #e0e0e0;
        min-width: 240px;
        max-width: 240px;
    }

    QLabel#SidebarTitle {
        font-size: 16px;
        font-weight: bold;
        padding: 15px 0 10px 20px;
    }

    QTreeWidget {
        background-color: transparent;
        border: none;
        font-size: 14px;
    }
    QTreeWidget::header { height: 0px; }
    QTreeWidget::item { padding: 8px 10px; }
    QTreeWidget::item:!root { padding-left: 20px; }

    QTreeWidget::item:selected {
        background-color: #e0e0e0;
        color: #000;
        border-radius: 6px;
    }

    QWidget#MainArea {
        padding: 10px 25px;
    }

    QLineEdit#SearchBar {
        background-color: #f0f0f0;
        border: none;
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 14px;
    }

    QLabel#FolderTitle {
        font-size: 28px;
        font-weight: bold;
        padding-top: 20px;
        padding-bottom: 15px;
    }

    QListWidget {
        background-color: #ffffff;
        border: none;
        outline: none;
    }

    QListWidget::item {
    background-color: #ffffff;
    border: 2px solid transparent;
    border-radius: 8px;
    padding: 4px;
    }

    QListWidget::item:hover {
        border: 2px solid #cccccc;
        background-color: #f0f0f0;
    }

    QListWidget::item:selected {
        border: 2px solid #007aff;
        background-color: #e8f0fe;
    }

    QWidget#EmptyPage QLabel { font-size: 16px; color: #888; }

    QPushButton#AddImagesButton {
        background-color: #007aff;
        color: white;
        font-size: 14px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        margin-top: 15px;
        max-width: 200px;
    }
    QPushButton#AddImagesButton:hover { background-color: #005ec4; }

    QPushButton#AddFolderButton {
        background-color: #e8e8e8;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        margin: 10px;
        padding: 10px;
    }
    QPushButton#AddFolderButton:hover { background-color: #dcdcdc; }
"""


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print(f"Error in worker thread: {e}")
            self.signals.error.emit(str(e))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class ThumbnailWorkerSignals(QObject):
    thumbnail_ready = Signal(str, QIcon)
    finished = Signal()


class ThumbnailWorker(QRunnable):
    def __init__(self, paths, icon_size):
        super().__init__()
        self.signals = ThumbnailWorkerSignals()
        self.paths = paths
        self.icon_size = icon_size
        self._is_stopped = False

    @Slot()
    def run(self):
        for path in self.paths:
            if self._is_stopped:
                break

            try:
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    continue

                scaled_pixmap = pixmap.scaled(
                    self.icon_size,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation
                )

                x = (scaled_pixmap.width() - self.icon_size.width()) / 2
                y = (scaled_pixmap.height() - self.icon_size.height()) / 2
                cropped_pixmap = scaled_pixmap.copy(
                    int(x), int(y),
                    self.icon_size.width(),
                    self.icon_size.height()
                )

                icon = QIcon(cropped_pixmap)
                self.signals.thumbnail_ready.emit(path, icon)

            except Exception as e:
                print(f"Error loading thumbnail {path}: {e}")

        if not self._is_stopped:
            self.signals.finished.emit()

    def stop(self):
        self._is_stopped = True


class SnapSearchWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Snap Search")
        self.setGeometry(100, 100, 1200, 800)

        self.threadpool = QThreadPool()
        print(f"Running with a thread pool of max {self.threadpool.maxThreadCount()} threads.")

        self.thumbnail_worker = None
        self.current_item_map = {}

        self.mainWidget = QWidget(objectName="MainWidget")
        self.mainLayout = QHBoxLayout(self.mainWidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.setCentralWidget(self.mainWidget)

        self.initUI()
        self.setStyleSheet(APP_STYLESHEET)

        self.load_folders_into_sidebar()
        self.tree.setCurrentItem(self.bootItem)

    def initUI(self):
        style = QApplication.style()
        self.folderIcon = style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        self.driveIcon = style.standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)
        self.homeIcon = style.standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon)
        self.browseIcon = style.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)

        self.sidebar = QWidget(objectName="Sidebar")
        sidebarLayout = QVBoxLayout(self.sidebar)
        sidebarLayout.setContentsMargins(0, 0, 0, 0)
        sidebarLayout.setSpacing(0)

        title = QLabel("Snap Search", objectName="SidebarTitle")
        sidebarLayout.addWidget(title)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        sidebarLayout.addWidget(self.tree, 1)

        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.showTreeContextMenu)

        discoverRoot = QTreeWidgetItem(self.tree, ["Discover"])
        self.homeItem = QTreeWidgetItem(discoverRoot, ["Home"])
        self.homeItem.setIcon(0, self.homeIcon)
        self.browseItem = QTreeWidgetItem(discoverRoot, ["Browse"])
        self.browseItem.setIcon(0, self.browseIcon)

        self.volumesRoot = QTreeWidgetItem(self.tree, ["Volumes"])
        self.volumesRoot.setIcon(0, self.driveIcon)

        self.bootItem = QTreeWidgetItem(self.volumesRoot, ["Macintosh HD"])
        self.bootItem.setIcon(0, self.driveIcon)
        self.bootItem.setData(0, Qt.UserRole, "/")

        try:
            volumes_path = "/Volumes"
            all_volumes = os.listdir(volumes_path)
            boot_drive_real_path = os.path.realpath("/")

            for volume_name in all_volumes:
                full_path = os.path.join(volumes_path, volume_name)
                if (volume_name.startswith('.') or
                        volume_name == "Macintosh HD" or
                        volume_name == "Recovery" or
                        os.path.realpath(full_path) == boot_drive_real_path):
                    continue
                if os.path.isdir(full_path):
                    item = QTreeWidgetItem(self.volumesRoot, [volume_name])
                    item.setData(0, Qt.UserRole, full_path)
                    item.setIcon(0, self.driveIcon)
        except Exception as e:
            print(f"Could not scan /Volumes: {e}")

        self.foldersRoot = QTreeWidgetItem(self.tree, ["Folders"])
        self.foldersRoot.setIcon(0, self.folderIcon)
        self.tree.expandAll()

        self.addFolderButton = QPushButton("+ New Folder", objectName="AddFolderButton")
        sidebarLayout.addWidget(self.addFolderButton)

        self.mainArea = QWidget(objectName="MainArea")
        mainLayout = QVBoxLayout(self.mainArea)
        mainLayout.setSpacing(0)
        mainLayout.setContentsMargins(10, 10, 25, 10)

        self.searchBar = QLineEdit(objectName="SearchBar")
        self.searchBar.setPlaceholderText("Search your photos")
        self.searchBar.setClearButtonEnabled(True)
        mainLayout.addWidget(self.searchBar)

        self.folderTitle = QLabel("Home", objectName="FolderTitle")
        mainLayout.addWidget(self.folderTitle)

        self.stackedWidget = QStackedWidget()
        mainLayout.addWidget(self.stackedWidget, 1)

        # --- Page 0: Image Grid ---
        self.imageGrid = QListWidget()
        self.imageGrid.setViewMode(QListWidget.IconMode)
        self.imageGrid.setIconSize(QSize(195, 195))
        self.imageGrid.setGridSize(QSize(220, 220))
        self.imageGrid.setMovement(QListView.Static)
        self.imageGrid.setResizeMode(QListWidget.Adjust)
        self.imageGrid.setMovement(QListWidget.Static)
        self.imageGrid.setSpacing(10)
        self.imageGrid.setWordWrap(False)
        self.imageGrid.setUniformItemSizes(True)

        # --- Page 1: Empty Folder ---
        self.emptyPage = QWidget(objectName="EmptyPage")
        emptyLayout = QVBoxLayout(self.emptyPage)
        emptyLayout.setAlignment(Qt.AlignCenter)
        self.emptyLabel = QLabel("This folder is empty.")
        self.addImagesButton = QPushButton("Add Images to Folder", objectName="AddImagesButton")
        emptyLayout.addWidget(self.emptyLabel)
        emptyLayout.addWidget(self.addImagesButton)
        emptyLayout.addStretch()

        self.stackedWidget.addWidget(self.imageGrid)
        self.stackedWidget.addWidget(self.emptyPage)

        self.mainLayout.addWidget(self.sidebar)
        self.mainLayout.addWidget(self.mainArea, 1)

        self.tree.currentItemChanged.connect(self.onFolderChanged)
        self.addFolderButton.clicked.connect(self.onAddNewFolder)
        self.addImagesButton.clicked.connect(self.onAddImagesToEmptyFolder)
        self.searchBar.returnPressed.connect(self.onSearch)

    def load_folders_into_sidebar(self, volume_filter_path=None):
        self.foldersRoot.takeChildren()

        managed_folders = folderManager.load_folders()

        for folder_path in managed_folders:
            if volume_filter_path:
                try:
                    norm_folder = os.path.normpath(folder_path)
                    norm_volume = os.path.normpath(volume_filter_path)

                    if norm_volume == "/":
                        if not norm_folder.startswith("/"):
                            continue
                        if norm_folder.startswith("/Volumes/"):
                            continue
                    elif not norm_folder.startswith(norm_volume):
                        continue
                except Exception as e:
                    print(f"Error checking path {folder_path} against {volume_filter_path}: {e}")
                    continue

            folder_name = os.path.basename(folder_path)
            item = QTreeWidgetItem(self.foldersRoot, [folder_name])
            item.setData(0, Qt.UserRole, folder_path)
            item.setIcon(0, self.folderIcon)

        self.foldersRoot.setExpanded(True)

    def loadImagesToGrid(self, folder_path, filenames):
        """LAZY LOADING: Populates grid with empty items, then loads thumbnails in background."""

        if self.thumbnail_worker:
            try:
                self.thumbnail_worker.signals.thumbnail_ready.disconnect(self.onThumbnailReady)
            except RuntimeError:
                pass
            self.thumbnail_worker.stop()

        self.imageGrid.clear()
        self.current_item_map.clear()

        paths_to_load = []
        for name in filenames:
            full_path = os.path.join(folder_path, name)
            item = QListWidgetItem()
            item.setSizeHint(QSize(210, 210))  # Set item size hint

            self.imageGrid.addItem(item)
            paths_to_load.append(full_path)
            self.current_item_map[full_path] = item

        if paths_to_load:
            icon_size = self.imageGrid.iconSize()
            self.thumbnail_worker = ThumbnailWorker(paths_to_load, icon_size)
            self.thumbnail_worker.signals.thumbnail_ready.connect(self.onThumbnailReady)
            self.threadpool.start(self.thumbnail_worker)

    def check_and_load_folder(self, folder_path):
        """
        A single function to run in a background thread.
        1. Runs the "smart" pipeline to find/process new images.
        2. Loads the (now updated) index and data into the cache.
        Returns 'True' on success, 'False' if the folder is empty.
        """
        print(f"Checking for updates in: {folder_path}")
        pipeline_main.generate_captions_and_embeddings(folder_path)
        print(f"Loading '{folder_path}' into memory cache...")
        success = build_index.load_folder_into_memory(folder_path)
        return success

    def run_background_task(self, task_function, *args):
        worker = Worker(task_function, *args)
        worker.signals.finished.connect(self.on_task_finished)
        worker.signals.error.connect(self.on_task_error)
        self.threadpool.start(worker)

    # --- NEW HELPER FUNCTION ---
    def start_folder_scan(self, item):
        """
        Starts the background task for scanning/loading a folder.
        This is called by both onFolderChanged and onRefreshFolder.
        """
        if not item or item.parent() != self.foldersRoot:
            return

        folder_path = item.data(0, Qt.UserRole)
        folder_name = item.text(0)

        print(f"Scan triggered for: {folder_path}")

        # 1. Set the UI to a "loading" state
        self.searchBar.clear()
        self.folderTitle.setText(f"Scanning '{folder_name}'...")
        self.stackedWidget.setCurrentIndex(1)  # Show empty page
        self.emptyLabel.setText(f"Scanning '{folder_name}' for new images...")

        # 2. Stop any pending thumbnail loads
        if self.thumbnail_worker:
            try:
                self.thumbnail_worker.signals.thumbnail_ready.disconnect(self.onThumbnailReady)
            except RuntimeError:
                pass
            self.thumbnail_worker.stop()

        # 3. Start the background task
        worker = Worker(self.check_and_load_folder, folder_path)

        # 4. Connect the worker's signals to our slots
        worker.signals.result.connect(self.on_folder_load_complete)
        worker.signals.error.connect(self.on_task_error)

        self.threadpool.start(worker)

    # --- SLOTS (Functions that respond to signals) ---

    @Slot(str, QIcon)
    def onThumbnailReady(self, path, icon):
        item = self.current_item_map.get(path)
        if item:
            item.setIcon(icon)

    # --- MODIFIED ---
    @Slot(QTreeWidgetItem, QTreeWidgetItem)
    def onFolderChanged(self, currentItem, previousItem):
        """
        This is the MAIN controller for the app.
        Called EVERY time the user clicks a different item in the sidebar tree.
        """
        if not currentItem:
            return

        parent = currentItem.parent()
        folder_path = currentItem.data(0, Qt.UserRole)
        folder_name = currentItem.text(0)

        # Set the title (the scan function will update it to "Scanning...")
        self.folderTitle.setText(folder_name)

        if parent != self.foldersRoot:
            self.searchBar.clear()

        if parent == self.foldersRoot:
            # --- 1. A MANAGED FOLDER was clicked ---
            # Instead of running all the logic here, just call the helper
            self.start_folder_scan(currentItem)

        elif parent == self.volumesRoot:
            # --- 2. A VOLUME was clicked ---
            print(f"Volume clicked: {folder_path}")
            self.load_folders_into_sidebar(volume_filter_path=folder_path)

            if self.foldersRoot.childCount() > 0:
                firstChild = self.foldersRoot.child(0)
                self.tree.setCurrentItem(firstChild)
            else:
                self.emptyLabel.setText(f"Showing folders on '{folder_name}'.\nSelect a folder or create a new one.")
                self.stackedWidget.setCurrentIndex(1)

        else:
            # --- 3. "Home", "Browse", or a header was clicked ---
            self.load_folders_into_sidebar(volume_filter_path=None)
            self.emptyLabel.setText("Select a folder to get started.")
            self.stackedWidget.setCurrentIndex(1)

    # --- NEW SLOT ---
    @Slot(object)
    def on_folder_load_complete(self, success):
        """
        Called when the background 'check_and_load_folder' task is done.
        This function is now responsible for populating the grid.
        """
        currentItem = self.tree.currentItem()
        if not currentItem: return
        folder_path = currentItem.data(0, Qt.UserRole)
        folder_name = currentItem.text(0)

        # Restore the title (it was "Scanning...")
        self.folderTitle.setText(folder_name)

        if not success:
            print(f"Folder '{folder_name}' is empty or failed to load.")
            self.emptyLabel.setText(f"'{folder_name}' has no indexed images.")
            self.stackedWidget.setCurrentIndex(1)
            return

        print("Folder scan and load complete. Populating grid.")

        all_indexed_ids = build_index.IDS_CACHE
        all_filenames = [
            build_index.DATA_CACHE[img_id]['filename']
            for img_id in all_indexed_ids
            if img_id in build_index.DATA_CACHE
        ]

        self.loadImagesToGrid(folder_path, all_filenames)
        self.stackedWidget.setCurrentIndex(0)

    @Slot()
    def onSearch(self):
        query = self.searchBar.text()
        currentItem = self.tree.currentItem()
        if not currentItem: return

        folder_path = currentItem.data(0, Qt.UserRole)
        if not folder_path:
            print("Cannot search a non-folder item.")
            return

        if not query:
            self.onFolderChanged(currentItem, None)
            return

        print(f"Searching for '{query}' in '{folder_path}'...")
        matching_filenames = build_index.searchSimilaritems(query)

        self.folderTitle.setText(f"{len(matching_filenames)} results for '{query}'")

        if matching_filenames:
            self.loadImagesToGrid(folder_path, matching_filenames)
            self.stackedWidget.setCurrentIndex(0)
        else:
            self.emptyLabel.setText(f"No results found for '{query}'.")
            self.stackedWidget.setCurrentIndex(1)

    @Slot()
    def onAddNewFolder(self):
        start_path = str(Path.home())
        currentItem = self.tree.currentItem()

        if currentItem and currentItem.parent() == self.volumesRoot:
            start_path = currentItem.data(0, Qt.UserRole)

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select or Create a Folder to Manage",
            start_path
        )
        if not folder_path:
            return

        folderManager.add_folder(folder_path)
        self.load_folders_into_sidebar()

        for i in range(self.foldersRoot.childCount()):
            child = self.foldersRoot.child(i)
            if child.data(0, Qt.UserRole) == folder_path:
                self.tree.setCurrentItem(child)
                break

        print(f"Starting background processing for new folder: {folder_path}")
        self.folderTitle.setText(f"Processing '{os.path.basename(folder_path)}'...")
        self.run_background_task(pipeline_main.generate_captions_and_embeddings, folder_path)

    @Slot()
    def onAddImagesToEmptyFolder(self):
        currentItem = self.tree.currentItem()
        if not currentItem: return

        folder_path = currentItem.data(0, Qt.UserRole)
        if not folder_path:
            print("Cannot add images to a non-folder item.")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images to add",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.heic *.webp)"
        )
        if not files:
            return

        print(f"Adding {len(files)} new files to {folder_path}...")
        for f in files:
            new_name = os.path.basename(f)
            new_path = os.path.join(folder_path, new_name)
            try:
                shutil.copy(f, new_path)
            except Exception as e:
                print(f"Error copying file {f}: {e}")

        print(f"Starting background processing for new images in: {folder_path}")
        self.folderTitle.setText(f"Processing '{os.path.basename(folder_path)}'...")
        self.run_background_task(pipeline_main.generate_captions_and_embeddings, folder_path)

    # --- MODIFIED ---
    @Slot(QPoint)
    def showTreeContextMenu(self, point):
        item = self.tree.itemAt(point)
        if not item: return

        if item.parent() == self.foldersRoot:
            menu = QMenu()

            # --- NEW REFRESH ACTION ---
            refresh_action = QAction("Refresh")
            refresh_action.triggered.connect(lambda: self.onRefreshFolder(item))
            menu.addAction(refresh_action)

            menu.addSeparator()

            # --- MODIFIED DELETE ACTION ---
            delete_action = QAction("Delete Folder")
            delete_action.triggered.connect(lambda: self.confirmDeleteFolder(item))
            menu.addAction(delete_action)

            menu.exec(self.tree.mapToGlobal(point))

    # --- NEW SLOT ---
    @Slot()
    def onRefreshFolder(self, item):
        """
        Called by the 'Refresh' context menu action for a specific item.
        Forces a re-scan of that folder.
        """
        # 1. Set the view to this item (so the user sees what's being refreshed)
        self.tree.setCurrentItem(item)
        # 2. Call our main scan function
        self.start_folder_scan(item)

    # --- MODIFIED ---
    @Slot()
    def confirmDeleteFolder(self, item):  # <-- Added 'item' argument
        """
        Shows the confirmation pop-up before deleting.
        'item' is passed in from the context menu.
        """
        if not item or item.parent() != self.foldersRoot:
            return

        folder_name = item.text(0)  # Use 'item'
        folder_path = item.data(0, Qt.UserRole)  # Use 'item'

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to permanently delete:\n\n"
            f"<b>{folder_name}</b>\n\n"
            f"This will delete the folder and all its images from your hard drive. "
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            print(f"Starting deletion for: {folder_path}")
            self.run_background_task(folderManager.delete_folder_permanently, folder_path)

    @Slot()
    def on_task_finished(self):
        print("Background task finished.")

        self.load_folders_into_sidebar(volume_filter_path=None)
        self.tree.setCurrentItem(self.homeItem)
        self.onFolderChanged(self.homeItem, None)

        QMessageBox.information(self, "Task Complete", "The operation finished successfully.")

    @Slot(str)
    def on_task_error(self, error_message):
        print(f"Background task failed: {error_message}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")

        # Reset view to home
        self.onFolderChanged(self.homeItem, None)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnapSearchWindow()
    window.show()
    sys.exit(app.exec())