"""Command-line interface for the Corporate LLM system."""
import argparse
from corporate_llm import CorporateLLM


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Corporate LLM - Question answering on internal documents'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing index before indexing'
    )
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', type=str, help='The question to ask')
    ask_parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Number of relevant chunks to retrieve'
    )
    ask_parser.add_argument(
        '--no-context',
        action='store_true',
        help='Answer without using document context'
    )
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear the document index')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Initialize the system
    llm_system = CorporateLLM()
    
    if args.command == 'index':
        llm_system.index_documents(clear_existing=args.clear)
    
    elif args.command == 'ask':
        result = llm_system.ask(
            question=args.question,
            top_k=args.top_k,
            use_context=not args.no_context
        )
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        
        if result['sources']:
            print("\n" + "="*80)
            print("SOURCES:")
            print("="*80)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']}")
        print("="*80 + "\n")
    
    elif args.command == 'stats':
        stats = llm_system.get_stats()
        print("\n" + "="*80)
        print("SYSTEM STATISTICS:")
        print("="*80)
        print(f"Total indexed chunks: {stats['total_chunks']}")
        print(f"Collection name: {stats['collection_name']}")
        print("="*80 + "\n")
    
    elif args.command == 'clear':
        confirm = input("Are you sure you want to clear the index? (yes/no): ")
        if confirm.lower() == 'yes':
            llm_system.clear_index()
            print("Index cleared successfully.")
        else:
            print("Operation cancelled.")
    
    elif args.command == 'interactive':
        print("\n" + "="*80)
        print("CORPORATE LLM - Interactive Mode")
        print("="*80)
        print("Type 'quit' or 'exit' to leave interactive mode")
        print("Type 'stats' to see system statistics")
        print("="*80 + "\n")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    stats = llm_system.get_stats()
                    print(f"\nTotal indexed chunks: {stats['total_chunks']}")
                    continue
                
                if not question:
                    continue
                
                result = llm_system.ask(question)
                
                print("\n" + "-"*80)
                print("ANSWER:")
                print("-"*80)
                print(result['answer'])
                
                if result['sources']:
                    print("\n" + "-"*80)
                    print("SOURCES:")
                    print("-"*80)
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['filename']}")
                print("-"*80)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
