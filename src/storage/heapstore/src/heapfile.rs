use crate::page::Page;
use common::prelude::*;
use common::PAGE_SIZE;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, RwLock};

/// Byte size of a single page on disk, cast to u64 for use in seek offset calculations
const PAGE_SIZE_U64: u64 = PAGE_SIZE as u64;

/// The struct for a heap file.
///
/// Uses interior mutability (`Arc<RwLock<File>>`) to allow concurrent reads and writes
/// without requiring a mutable reference to the HeapFile itself.
///
/// Note: HeapFile cannot be serialized — callers should persist the file path and
/// container_id separately to reconstruct it on restart.
pub(crate) struct HeapFile {
    /// The underlying file handle, shared and protected by a read-write lock
    pub file: Arc<RwLock<File>>,
    /// The container this heap file belongs to
    pub container_id: ContainerId,
    /// Number of pages read from this file (used for profiling)
    pub read_count: AtomicU16,
    /// Number of pages written to this file (used for profiling)
    pub write_count: AtomicU16,
}

impl HeapFile {
    /// Opens or creates a heap file at the given path for the given container.
    /// Returns an error if the file cannot be opened due to permissions, missing
    /// directories, or other OS-level issues.
    pub(crate) fn new(file_path: PathBuf, container_id: ContainerId) -> Result<Self, CrustyError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_path)
            .map_err(|error| {
                CrustyError::CrustyError(format!(
                    "Cannot open or create heap file: {} {:?}",
                    file_path.to_string_lossy(),
                    error
                ))
            })?;

        Ok(HeapFile {
            file: Arc::new(RwLock::new(file)),
            container_id,
            read_count: AtomicU16::new(0),
            write_count: AtomicU16::new(0),
        })
    }

    /// Returns the number of pages currently stored in this heap file.
    /// Computed from the file size divided by PAGE_SIZE.
    pub fn num_pages(&self) -> PageId {
        let file = match self.file.read() {
            Ok(f) => f,
            Err(_) => return 0,
        };

        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        (file_len / PAGE_SIZE_U64) as PageId
    }

    /// Reads and deserializes the page at the given page_id from disk.
    /// Returns an error if the page_id is out of range or the read fails.
    ///
    /// Note: Seeking requires a mutable file handle, so this clones the file
    /// descriptor rather than holding the read lock across the seek+read.
    pub(crate) fn read_page_from_file(&self, page_id: PageId) -> Result<Page, CrustyError> {
        #[cfg(feature = "profile")]
        {
            self.read_count.fetch_add(1, Ordering::Relaxed);
        }

        let file = self.file.read().map_err(|error| {
            CrustyError::CrustyError(format!("Failed to acquire read lock on heap file: {:?}", error))
        })?;

        // Clone the file descriptor so we can seek without holding the lock
        let mut file_handle = file.try_clone().expect("OS failed to clone file handle");
        drop(file);

        let byte_offset = page_id as u64 * PAGE_SIZE_U64;
        let mut buffer = [0u8; PAGE_SIZE];

        file_handle
            .seek(SeekFrom::Start(byte_offset))
            .map_err(|e| CrustyError::CrustyError(format!("Seek failed for page {}: {:?}", page_id, e)))?;

        file_handle
            .read_exact(&mut buffer)
            .map_err(|e| CrustyError::CrustyError(format!("Read failed for page {}: {:?}", page_id, e)))?;

        Ok(Page::from_bytes(buffer))
    }

    /// Serializes and writes the given page to its position in the heap file.
    /// The write position is determined by the page's own page_id.
    /// This can write to an existing page (update) or extend the file (new page).
    pub(crate) fn write_page_to_file(&self, page: &Page) -> Result<(), CrustyError> {
        trace!(
            "Writing page {} to container {}",
            page.get_page_id(),
            self.container_id
        );

        #[cfg(feature = "profile")]
        {
            self.write_count.fetch_add(1, Ordering::Relaxed);
        }

        let page_id = page.get_page_id();
        let page_data = page.to_bytes();
        let byte_offset = page_id as u64 * PAGE_SIZE_U64;

        let mut file = self.file.write().map_err(|error| {
            CrustyError::CrustyError(format!("Failed to acquire write lock on heap file: {:?}", error))
        })?;

        file.seek(SeekFrom::Start(byte_offset))
            .map_err(|e| CrustyError::CrustyError(format!("Seek failed for page {}: {:?}", page_id, e)))?;

        file.write_all(page_data)
            .map_err(|e| CrustyError::CrustyError(format!("Write failed for page {}: {:?}", page_id, e)))?;

        Ok(())
    }

    /// Returns the total size of the heap file in bytes.
    pub(crate) fn get_file_size(&self) -> usize {
        let file = match self.file.read() {
            Ok(f) => f,
            Err(_) => return 0,
        };

        file.metadata().map(|m| m.len()).unwrap_or(0) as usize
    }
}

#[cfg(test)]
#[allow(unused_must_use)]
mod test {
    use crate::page::HeapPage;

    use super::*;
    use common::testutil::*;
    use temp_testdir::TempDir;

    #[test]
    fn hs_hf_insert() {
        init();

        //Create a temp file
        let f = gen_random_test_sm_dir();
        let tdir = TempDir::new(f, true);
        let mut f = tdir.to_path_buf();
        f.push(gen_rand_string(4));
        f.set_extension("hf");

        let mut hf = HeapFile::new(f.to_path_buf(), 0).expect("Unable to create HF for test");

        // Make a page and write
        let mut p0 = Page::new(0);
        let bytes = get_random_byte_vec(100);
        p0.add_value(&bytes);
        let bytes = get_random_byte_vec(100);
        p0.add_value(&bytes);
        let bytes = get_random_byte_vec(100);
        p0.add_value(&bytes);
        let p0_bytes = p0.to_bytes();

        hf.write_page_to_file(&p0);
        //check the page
        assert_eq!(1, hf.num_pages());
        let checkp0 = hf.read_page_from_file(0).unwrap();
        assert_eq!(p0_bytes, checkp0.to_bytes());

        //Add another page
        let mut p1 = Page::new(1);
        let bytes = get_random_byte_vec(100);
        p1.add_value(&bytes);
        let bytes = get_random_byte_vec(100);
        p1.add_value(&bytes);
        let bytes = get_random_byte_vec(100);
        p1.add_value(&bytes);
        let p1_bytes = p1.to_bytes();

        hf.write_page_to_file(&p1);

        assert_eq!(2, hf.num_pages());
        //Recheck page0
        let checkp0 = hf.read_page_from_file(0).unwrap();
        assert_eq!(p0_bytes, checkp0.to_bytes());

        //check page 1
        let checkp1 = hf.read_page_from_file(1).unwrap();
        assert_eq!(p1_bytes, checkp1.to_bytes());

        #[cfg(feature = "profile")]
        {
            assert_eq!(*hf.read_count.get_mut(), 3);
            assert_eq!(*hf.write_count.get_mut(), 2);
        }
    }
}