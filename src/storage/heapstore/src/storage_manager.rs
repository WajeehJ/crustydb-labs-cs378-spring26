use crate::heap_page::HeapPage;
use crate::heapfile::HeapFile;
use crate::heapfileiter::HeapFileIterator;
use crate::page::Page;
use common::prelude::*;
use common::storage_trait::StorageTrait;
use common::testutil::gen_random_test_sm_dir;
use common::PAGE_SIZE;
use env_logger::DEFAULT_WRITE_STYLE_ENV;
use std::collections::HashMap;
use std::path::{self, Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};
use std::{fs, num};
/// Directory name used for heapstore data files
pub const STORAGE_DIR: &str = "heapstore";

/// Filename suffix for heap file containers
const HEAP_FILE_EXTENSION: &str = "hf";

/// Filename for the serialized storage manager config, used for persistence across restarts
const PERSIST_CONFIG_FILENAME: &str = "storage_manager";

/// Maps a ContainerId to its in-memory HeapFile handle
pub(crate) type ContainerMap = Arc<RwLock<HashMap<ContainerId, Arc<HeapFile>>>>;

/// Maps a ContainerId to the path of its backing heap file on disk
pub(crate) type ContainerPathMap = Arc<RwLock<HashMap<ContainerId, Arc<PathBuf>>>>;

/// The StorageManager is responsible for managing heap file containers.
/// It maps container IDs to HeapFile instances and persists this mapping across restarts.
#[derive(Serialize, Deserialize)]
pub struct StorageManager {
    /// Path to the directory where heap files and metadata are stored
    pub storage_dir: PathBuf,
    /// If true, this is a temporary SM used for testing; storage is deleted on drop
    is_temp: bool,
    /// Persisted mapping from container ID to heap file path
    pub(crate) cid_path_map: ContainerPathMap,
    /// In-memory mapping from container ID to open HeapFile handle (not serialized)
    #[serde(skip)]
    pub(crate) cid_heapfile_map: ContainerMap,
}

/// HeapStore-specific StorageManager functions that operate directly on HeapFiles
impl StorageManager {
    /// Reads and returns the page with the given page_id from the specified container.
    /// Returns None if the page cannot be read.
    pub(crate) fn get_page(
        &self,
        container_id: ContainerId,
        page_id: PageId,
        _tid: TransactionId,
        _perm: Permissions,
        _pin: bool,
    ) -> Option<Page> {
        let guard = self.cid_heapfile_map.read().unwrap();
        let container = guard.get(&container_id).unwrap();
        let page = container
            .read_page_from_file(page_id)
            .expect("Failed to read page from heap file");
        Some(page)
    }

    /// Writes the given page to the specified container's heap file.
    pub(crate) fn write_page(
        &self,
        container_id: ContainerId,
        page: &Page,
        _tid: TransactionId,
    ) -> Result<(), CrustyError> {
        let guard = self.cid_heapfile_map.read().unwrap();
        let container = guard.get(&container_id).unwrap();
        container
            .write_page_to_file(page)
            .expect("Failed to write page to heap file");
        Ok(())
    }

    /// Returns the number of pages currently allocated in the given container.
    fn get_num_pages(&self, container_id: ContainerId) -> PageId {
        let guard = self.cid_heapfile_map.read().unwrap();
        let container = guard.get(&container_id).unwrap();
        container.num_pages()
    }

    /// Test utility function for counting reads and writes served by the heap file.
    /// Can return 0,0 for invalid container_ids
    #[allow(dead_code)]
    pub(crate) fn get_hf_read_write_count(&self, container_id: ContainerId) -> (u16, u16) {
        panic!("TODO milestone hs");
    }

    /// Returns a debug string representation of a page, used for testing.
    pub fn get_page_debug(&self, container_id: ContainerId, page_id: PageId) -> String {
        match self.get_page(
            container_id,
            page_id,
            TransactionId::new(),
            Permissions::ReadOnly,
            false,
        ) {
            Some(page) => format!("{:?}", page),
            None => String::new(),
        }
    }

    /// Returns the heap file path for a given container ID.
    /// Heap files are stored as `<storage_dir>/<container_id>.hf`.
    fn container_file_path(&self, container_id: ContainerId) -> PathBuf {
        self.storage_dir
            .join(format!("{}.{}", container_id, HEAP_FILE_EXTENSION))
    }

    /// Retrieves a cloned Arc to the HeapFile for the given container.
    /// Panics if the container does not exist.
    fn get_container(&self, container_id: ContainerId) -> Arc<HeapFile> {
        let guard = self.cid_heapfile_map.read().unwrap();
        guard.get(&container_id).cloned().unwrap()
    }
}

/// Implementation of the StorageTrait for StorageManager
impl StorageTrait for StorageManager {
    type ValIterator = HeapFileIterator;

    /// Creates a StorageManager backed by the given directory.
    /// If a persisted config exists in the directory, it is loaded and HeapFiles are reopened.
    /// Otherwise a fresh StorageManager is initialized.
    fn new(storage_dir: &Path) -> Self {
        let config_path = storage_dir.join(PERSIST_CONFIG_FILENAME);

        if config_path.exists() {
            debug!("Loading storage manager from config file {:?}", config_path);

            let reader =
                fs::File::open(config_path).expect("Failed to open persisted config file");
            let persisted_sm: StorageManager =
                serde_json::from_reader(reader).expect("Failed to deserialize storage manager");

            let mut heapfile_map: HashMap<ContainerId, Arc<HeapFile>> = HashMap::new();
            let mut path_map: HashMap<ContainerId, Arc<PathBuf>> = HashMap::new();

            let old_path_map = persisted_sm.cid_path_map.read().unwrap();
            for (container_id, file_path) in old_path_map.iter() {
                let heap_file = HeapFile::new(file_path.as_ref().clone(), *container_id)
                    .expect("Failed to reopen heap file from persisted path");
                path_map.insert(*container_id, Arc::new(file_path.as_ref().clone()));
                heapfile_map.insert(*container_id, Arc::new(heap_file));
            }

            StorageManager {
                storage_dir: storage_dir.to_path_buf(),
                cid_heapfile_map: Arc::new(RwLock::new(heapfile_map)),
                cid_path_map: Arc::new(RwLock::new(path_map)),
                is_temp: false,
            }
        } else {
            debug!("Creating new storage manager in directory {:?}", storage_dir);
            fs::create_dir_all(storage_dir).expect("Failed to create storage directory");

            StorageManager {
                storage_dir: storage_dir.to_path_buf(),
                cid_heapfile_map: Arc::new(RwLock::new(HashMap::new())),
                cid_path_map: Arc::new(RwLock::new(HashMap::new())),
                is_temp: false,
            }
        }
    }

    /// Creates a temporary StorageManager for use in tests.
    /// Uses a randomly generated directory and sets is_temp=true so storage is cleaned up on drop.
    fn new_test_sm() -> Self {
        let storage_dir = gen_random_test_sm_dir();
        debug!("Creating temporary storage manager at {:?}", storage_dir);
        fs::create_dir_all(&storage_dir).expect("Failed to create temp storage directory");

        StorageManager {
            storage_dir: storage_dir.to_path_buf(),
            cid_heapfile_map: Arc::new(RwLock::new(HashMap::new())),
            cid_path_map: Arc::new(RwLock::new(HashMap::new())),
            is_temp: true,
        }
    }

    /// Inserts a value into the given container.
    /// Scans existing pages for one with enough space; creates a new page if none is found.
    /// Returns the ValueId (container, page, slot) where the value was stored.
    fn insert_value(
        &self,
        container_id: ContainerId,
        value: Vec<u8>,
        _tid: TransactionId,
    ) -> ValueId {
        if value.len() > PAGE_SIZE {
            panic!("Cannot insert a value larger than PAGE_SIZE");
        }

        let container = self.get_container(container_id);
        let num_pages = container.num_pages();

        // Try to insert into an existing page
        for page_id in 0..num_pages {
            let mut page = container
                .read_page_from_file(page_id)
                .expect("Failed to read page during insert");

            if let Some(slot_id) = page.add_value(&value) {
                container
                    .write_page_to_file(&page)
                    .expect("Failed to write page during insert");
                return ValueId::new_slot(container_id, page_id, slot_id);
            }
        }

        // No existing page had space — allocate a new one
        let mut new_page = Page::new(num_pages);
        let slot_id = new_page
            .add_value(&value)
            .expect("Failed to insert into a fresh page; value may exceed usable page space");
        container
            .write_page_to_file(&new_page)
            .expect("Failed to write new page during insert");

        ValueId::new_slot(container_id, num_pages, slot_id)
    }

    /// Inserts multiple values into a container by calling insert_value for each.
    /// Returns a vector of ValueIds in the same order as the input values.
    fn insert_values(
        &self,
        container_id: ContainerId,
        values: Vec<Vec<u8>>,
        tid: TransactionId,
    ) -> Vec<ValueId> {
        values
            .into_iter()
            .map(|value| self.insert_value(container_id, value, tid))
            .collect()
    }

    /// Deletes the value identified by the given ValueId.
    /// If the ValueId is not found, returns Ok(()) without error.
    fn delete_value(&self, id: ValueId, _tid: TransactionId) -> Result<(), CrustyError> {
        let page_id = id.page_id.unwrap();
        let slot_id = id.slot_id.unwrap();

        let container = self.get_container(id.container_id);

        let mut page = container
            .read_page_from_file(page_id)
            .expect("Failed to read page during delete");

        page.delete_value(slot_id);

        container
            .write_page_to_file(&page)
            .expect("Failed to write page after delete");

        Ok(())
    }

    /// Updates a value by deleting it and reinserting the new bytes.
    /// The returned ValueId may differ from the input if the value moved to a different page/slot.
    fn update_value(
        &self,
        value: Vec<u8>,
        id: ValueId,
        tid: TransactionId,
    ) -> Result<ValueId, CrustyError> {
        self.delete_value(id, tid)?;
        let new_value_id = self.insert_value(id.container_id, value, tid);
        Ok(new_value_id)
    }

    /// Creates a new container (HeapFile) for the given container_id.
    /// The heap file is created at `<storage_dir>/<container_id>.hf`.
    fn create_container(
        &self,
        container_id: ContainerId,
        _name: Option<String>,
        _container_type: common::ids::StateType,
        _dependencies: Option<Vec<ContainerId>>,
    ) -> Result<(), CrustyError> {
        let file_path = self.container_file_path(container_id);
        let heap_file = HeapFile::new(file_path.clone(), container_id)
            .expect("Failed to create heap file for new container");

        let mut heapfile_guard = self.cid_heapfile_map.write().unwrap();
        let mut path_guard = self.cid_path_map.write().unwrap();

        path_guard.insert(container_id, Arc::new(file_path));
        heapfile_guard.insert(container_id, Arc::new(heap_file));

        Ok(())
    }

    /// Convenience wrapper to create a base table container.
    fn create_table(&self, container_id: ContainerId) -> Result<(), CrustyError> {
        self.create_container(container_id, None, common::ids::StateType::BaseTable, None)
    }

    /// Removes the container from memory and deletes its heap file from disk.
    fn remove_container(&self, container_id: ContainerId) -> Result<(), CrustyError> {
        let mut heapfile_guard = self.cid_heapfile_map.write().unwrap();
        let mut path_guard = self.cid_path_map.write().unwrap();

        heapfile_guard.remove(&container_id);
        path_guard.remove(&container_id);

        let file_path = self.container_file_path(container_id);
        if file_path.exists() {
            fs::remove_file(&file_path).map_err(|e| {
                CrustyError::CrustyError(format!("Failed to delete container file: {}", e))
            })?;
        }

        Ok(())
    }

    /// Returns an iterator over all valid records in the given container.
    fn get_iterator(
        &self,
        container_id: ContainerId,
        tid: TransactionId,
        _perm: Permissions,
    ) -> Self::ValIterator {
        let container = self.get_container(container_id);
        HeapFileIterator::new(tid, container)
    }

    /// Returns an iterator starting from the given ValueId.
    fn get_iterator_from(
        &self,
        container_id: ContainerId,
        tid: TransactionId,
        _perm: Permissions,
        start: ValueId,
    ) -> Self::ValIterator {
        let container = self.get_container(container_id);
        HeapFileIterator::new_from(tid, container, start)
    }

    /// Retrieves the bytes stored at the given ValueId.
    /// Returns an error if the slot does not exist.
    fn get_value(
        &self,
        id: ValueId,
        _tid: TransactionId,
        _perm: Permissions,
    ) -> Result<Vec<u8>, CrustyError> {
        let container = self.get_container(id.container_id);

        let page = container
            .read_page_from_file(id.page_id.unwrap())
            .expect("Failed to read page during get_value");

        page.get_value(id.slot_id.unwrap())
            .ok_or_else(|| CrustyError::CrustyError("Value not found at given slot".to_string()))
    }

    fn get_storage_path(&self) -> &Path {
        &self.storage_dir
    }

    /// Clears all in-memory and on-disk state, then recreates the empty storage directory.
    fn reset(&self) -> Result<(), CrustyError> {
        fs::remove_dir_all(&self.storage_dir)?;
        fs::create_dir_all(&self.storage_dir).unwrap();

        let mut heapfile_guard = self.cid_heapfile_map.write().unwrap();
        let mut path_guard = self.cid_path_map.write().unwrap();
        heapfile_guard.clear();
        path_guard.clear();

        Ok(())
    }

    /// No-op: this StorageManager has no buffer pool or cache to clear.
    fn clear_cache(&self) {}

    /// Serializes the storage manager's container-to-path mapping to disk.
    /// Called on shutdown so the SM can be reconstructed on next startup.
    fn shutdown(&self) {
        debug!("serializing storage manager");
        let mut filename = self.storage_dir.clone();
        filename.push(PERSIST_CONFIG_FILENAME);
        serde_json::to_writer(
            fs::File::create(filename).expect("error creating file"),
            &self,
        )
        .expect("error serializing storage manager");
    }
}

/// Cleans up temporary storage directories when the StorageManager goes out of scope.
impl Drop for StorageManager {
    fn drop(&mut self) {
        if self.is_temp {
            debug!("Removing temp storage directory on drop {:?}", self.storage_dir);
            if let Err(e) = fs::remove_dir_all(&self.storage_dir) {
                println!("Error removing temp storage directory: {}", e);
            }
        }
    }
}


#[cfg(test)]
#[allow(unused_must_use)]
mod test {
    use super::*;
    use crate::storage_manager::StorageManager;
    use common::storage_trait::StorageTrait;
    use common::testutil::*;

    #[test]
    fn hs_sm_a_insert() {
        init();
        let sm = StorageManager::new_test_sm();
        let cid = 1;
        sm.create_table(cid);

        let bytes = get_random_byte_vec(40);
        let tid = TransactionId::new();

        let val1 = sm.insert_value(cid, bytes.clone(), tid);
        assert_eq!(1, sm.get_num_pages(cid));
        assert_eq!(0, val1.page_id.unwrap());
        assert_eq!(0, val1.slot_id.unwrap());

        let p1 = sm
            .get_page(cid, 0, tid, Permissions::ReadOnly, false)
            .unwrap();

        let val2 = sm.insert_value(cid, bytes, tid);
        assert_eq!(1, sm.get_num_pages(cid));
        assert_eq!(0, val2.page_id.unwrap());
        assert_eq!(1, val2.slot_id.unwrap());

        let p2 = sm
            .get_page(cid, 0, tid, Permissions::ReadOnly, false)
            .unwrap();
        assert_ne!(p1.to_bytes()[..], p2.to_bytes()[..]);
    }

    #[test]
    fn hs_sm_b_iter_small() {
        init();
        let sm = StorageManager::new_test_sm();
        let cid = 1;
        sm.create_table(cid);
        let tid = TransactionId::new();

        //Test one page
        let mut byte_vec: Vec<Vec<u8>> = vec![
            get_random_byte_vec(400),
            get_random_byte_vec(400),
            get_random_byte_vec(400),
        ];
        for val in &byte_vec {
            sm.insert_value(cid, val.clone(), tid);
        }
        let iter = sm.get_iterator(cid, tid, Permissions::ReadOnly);
        for (i, x) in iter.enumerate() {
            assert_eq!(byte_vec[i], x.0);
        }

        // Should be on two pages
        let mut byte_vec2: Vec<Vec<u8>> = vec![
            get_random_byte_vec(400),
            get_random_byte_vec(400),
            get_random_byte_vec(400),
            get_random_byte_vec(400),
        ];

        for val in &byte_vec2 {
            sm.insert_value(cid, val.clone(), tid);
        }
        byte_vec.append(&mut byte_vec2);

        let iter = sm.get_iterator(cid, tid, Permissions::ReadOnly);
        for (i, x) in iter.enumerate() {
            assert_eq!(byte_vec[i], x.0);
        }

        // Should be on 3 pages
        let mut byte_vec2: Vec<Vec<u8>> = vec![
            get_random_byte_vec(300),
            get_random_byte_vec(500),
            get_random_byte_vec(400),
        ];

        for val in &byte_vec2 {
            sm.insert_value(cid, val.clone(), tid);
        }
        byte_vec.append(&mut byte_vec2);

        let iter = sm.get_iterator(cid, tid, Permissions::ReadOnly);
        for (i, x) in iter.enumerate() {
            assert_eq!(byte_vec[i], x.0);
        }
    }

    #[test]
    #[ignore]
    fn hs_sm_b_iter_large() {
        init();
        let sm = StorageManager::new_test_sm();
        let cid = 1;

        sm.create_table(cid).unwrap();
        let tid = TransactionId::new();

        let vals = get_random_vec_of_byte_vec(1000, 40, 400);
        sm.insert_values(cid, vals, tid);
        let mut count = 0;
        for _ in sm.get_iterator(cid, tid, Permissions::ReadOnly) {
            count += 1;
        }
        assert_eq!(1000, count);
    }
}