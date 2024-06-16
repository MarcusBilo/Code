package main

import (
	"testing"
)

// go test -bench . -benchmem

func BenchmarkReplaceHTMLSpecialEntities1(b *testing.B) {
	input := "&amp;&lt;&gt;&quot;&lsquo;&rsquo;&tilde;&ndash;&mdash;&apos;"
	for i := 0; i < b.N; i++ {
		ReplaceHTMLSpecialEntities1(input)
	}
}

func BenchmarkReplaceHTMLSpecialEntities2(b *testing.B) {
	input := "&amp;&lt;&gt;&quot;&lsquo;&rsquo;&tilde;&ndash;&mdash;&apos;"
	for i := 0; i < b.N; i++ {
		ReplaceHTMLSpecialEntities2(input)
	}
}
