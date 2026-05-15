use crate::heap_page::HeapPage;
use crate::heap_page::HeapPageIntoIter;
use crate::heapfile::HeapFile;
use common::prelude::*;
use std::sync::Arc;

/// Iterator over all valid records in a HeapFile.
///
/// Iterates page by page, and within each page slot by slot.
/// Supports resuming mid-page via `new_from` using a starting `ValueId`.
pub struct HeapFileIterator {
    /// The heap file being iterated
    pub hf: Arc<HeapFile>,
    /// Transaction context for this iteration
    pub tid: TransactionId,
    /// Page index of the next page to load (None signals exhaustion)
    pub next_page_id: Option<PageId>,
    /// Slot to start from when first loading the current page (None means start from slot 0)
    pub resume_slot_id: Option<SlotId>,
    /// One past the last valid page index; iteration stops when next_page_id reaches this
    pub end_page_id: Option<PageId>,
    /// Iterator over the currently loaded page's slots
    pub current_page_iter: Option<HeapPageIntoIter>,
}

impl HeapFileIterator {
    /// Creates a new iterator that starts from the beginning of the heap file.
    pub(crate) fn new(tid: TransactionId, hf: Arc<HeapFile>) -> Self {
        HeapFileIterator {
            end_page_id: Some(hf.num_pages()),
            hf,
            tid,
            next_page_id: Some(0),
            resume_slot_id: None,
            current_page_iter: None,
        }
    }

    /// Creates a new iterator that resumes from the page and slot indicated by `start`.
    /// Useful for range scans or continuing after a checkpoint.
    pub(crate) fn new_from(tid: TransactionId, hf: Arc<HeapFile>, start: ValueId) -> Self {
        HeapFileIterator {
            end_page_id: Some(hf.num_pages()),
            hf,
            tid,
            next_page_id: start.page_id,
            resume_slot_id: start.slot_id,
            current_page_iter: None,
        }
    }
}

impl Iterator for HeapFileIterator {
    type Item = (Vec<u8>, ValueId);

    /// Advances the iterator, loading new pages as needed.
    /// Returns the next (value bytes, ValueId) pair, or None when exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        while self.next_page_id < self.end_page_id {
            // Load the next page into current_page_iter if we don't have one active
            if self.current_page_iter.is_none() {
                let page = self
                    .hf
                    .read_page_from_file(self.next_page_id?)
                    .ok()?;

                // If resuming mid-page, start at the saved slot; otherwise start from the beginning
                let page_iter = match self.resume_slot_id {
                    Some(slot_id) => HeapPageIntoIter::new_at_slot(page, slot_id),
                    None => page.into_iter(),
                };

                self.current_page_iter = Some(page_iter);
            }

            if let Some(ref mut page_iter) = self.current_page_iter {
                if let Some((bytes, slot_id)) = page_iter.next() {
                    let current_page_id = self.next_page_id.unwrap();
                    let value_id =
                        ValueId::new_slot(self.hf.container_id, current_page_id, slot_id);
                    return Some((bytes, value_id));
                }
            }

            // Current page is exhausted — advance to the next page
            self.current_page_iter = None;
            self.next_page_id = self.next_page_id.map(|page_id| page_id + 1);
            self.resume_slot_id = None;
        }

        None
    }
}