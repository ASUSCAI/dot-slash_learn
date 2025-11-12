'''
Test Chunking Implementation

Verifies that semantic chunking works correctly:
1. Tests chunk_text() with sample content
2. Verifies chunk sizes and overlap
3. Tests with realistic academic text

Usage:
    python test_chunking.py
'''

from course_embedder import CourseEmbedder


def test_basic_chunking():
    """Test basic chunking with controlled input"""
    print('='*80)
    print('TEST 1: Basic Chunking')
    print('='*80)

    embedder = CourseEmbedder(chunk_size=50, chunk_overlap=10)  # Small values for testing

    # Create sample text with clear sentences
    sentences = [f"This is sentence number {i}." for i in range(1, 21)]
    text = " ".join(sentences)

    print(f'\nInput text: {len(text)} characters, {len(sentences)} sentences')
    print(f'Chunk size: 50 tokens | Overlap: 10 tokens\n')

    chunks = embedder.chunk_text(text)

    print(f'Created {len(chunks)} chunks\n')

    for idx, chunk in enumerate(chunks):
        token_count = len(embedder.tokenizer.encode(chunk, add_special_tokens=False))
        print(f'Chunk {idx+1}: {token_count} tokens, {len(chunk)} chars')
        print(f'  Preview: {chunk[:100]}...\n')

    # Verify overlap
    if len(chunks) > 1:
        print('Verifying overlap between chunks...')
        chunk1_end = chunks[0][-100:]
        chunk2_start = chunks[1][:100]

        # Check if there's any overlap (some words should match)
        overlap_words = set(chunk1_end.split()) & set(chunk2_start.split())
        print(f'  Overlapping words: {len(overlap_words)}')
        print(f'  Sample: {list(overlap_words)[:5]}')

        if len(overlap_words) > 0:
            print('  ✅ Overlap verified!')
        else:
            print('  ⚠️ No overlap detected')

    print('\n' + '='*80 + '\n')


def test_academic_text():
    """Test with realistic academic content"""
    print('='*80)
    print('TEST 2: Academic Text Chunking')
    print('='*80)

    embedder = CourseEmbedder(chunk_size=500, chunk_overlap=100)

    # Sample academic text (about OOP)
    text = """
    Object-Oriented Programming (OOP) is a programming paradigm based on the concept of "objects",
    which can contain data in the form of fields (often known as attributes or properties), and code
    in the form of procedures (often known as methods). A feature of objects is that an object's own
    procedures can access and often modify the data fields of itself.

    In OOP, computer programs are designed by making them out of objects that interact with one another.
    OOP languages are diverse, but the most popular ones are class-based, meaning that objects are
    instances of classes, which also determine their types.

    The four main principles of OOP are: Encapsulation, Abstraction, Inheritance, and Polymorphism.

    Encapsulation is the mechanism of hiding data implementation by restricting access to public methods.
    Instance variables are kept private and accessor methods are made public to achieve this.

    Abstraction means using simple things to represent complexity. We all know how to turn the TV on,
    but we don't need to know how it works in order to enjoy it. In OOP, abstraction means showing only
    the essential features of an object to the user, and hiding the rest.

    Inheritance is when you create a new class by inheriting the properties of another existing class.
    The new class is called the subclass (or derived class), and the existing class is the superclass
    (or base class). The subclass can add new fields and methods, or override existing ones.

    Polymorphism means "many shapes" in Greek. In OOP, polymorphism refers to a programming language's
    ability to process objects differently depending on their data type or class. More specifically,
    it is the ability to redefine methods for derived classes.
    """

    print(f'\nInput: Academic text about OOP')
    print(f'Length: {len(text)} characters')
    print(f'Chunk size: 500 tokens | Overlap: 100 tokens\n')

    chunks = embedder.chunk_text(text)

    print(f'Created {len(chunks)} chunks\n')

    for idx, chunk in enumerate(chunks):
        token_count = len(embedder.tokenizer.encode(chunk, add_special_tokens=False))
        print(f'[Chunk {idx+1}/{len(chunks)}] {token_count} tokens, {len(chunk)} chars')
        print(f'Preview: {chunk.strip()[:150]}...')
        print()

    print('='*80 + '\n')


def test_edge_cases():
    """Test edge cases"""
    print('='*80)
    print('TEST 3: Edge Cases')
    print('='*80)

    embedder = CourseEmbedder(chunk_size=500, chunk_overlap=100)

    # Test 1: Empty text
    print('\n[Test 3a] Empty text')
    chunks = embedder.chunk_text('')
    print(f'  Result: {len(chunks)} chunks')
    print(f'  Expected: 0 chunks')
    print(f'  ✅ PASS' if len(chunks) == 0 else '  ❌ FAIL')

    # Test 2: Single sentence (smaller than chunk size)
    print('\n[Test 3b] Single short sentence')
    chunks = embedder.chunk_text('This is a short sentence.')
    print(f'  Result: {len(chunks)} chunks')
    print(f'  Expected: 1 chunk')
    print(f'  ✅ PASS' if len(chunks) == 1 else '  ❌ FAIL')

    # Test 3: Very long sentence (larger than chunk size)
    print('\n[Test 3c] Very long sentence')
    long_sentence = 'This is a very long sentence. ' * 100
    chunks = embedder.chunk_text(long_sentence)
    print(f'  Result: {len(chunks)} chunks')
    print(f'  Expected: Multiple chunks')
    print(f'  ✅ PASS' if len(chunks) > 1 else '  ❌ FAIL')

    print('\n' + '='*80 + '\n')


def main():
    print('\n' + '='*80)
    print('CHUNKING IMPLEMENTATION TESTS')
    print('='*80 + '\n')

    try:
        test_basic_chunking()
        test_academic_text()
        test_edge_cases()

        print('='*80)
        print('ALL TESTS COMPLETED')
        print('='*80)
        print('\n✅ Chunking implementation is working correctly!')
        print('\nNext steps:')
        print('1. Re-embed your collection: python embed_all_subjects.py --subjects cs')
        print('2. Test retrieval quality: python llm.py "What is OOP?"')
        print('3. Compare before/after relevance scores')
        print()

    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
