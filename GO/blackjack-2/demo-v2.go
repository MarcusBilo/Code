package main

import (
	"fmt"
	tea "github.com/charmbracelet/bubbletea"
	"log"
	"math/rand"
)

type model struct {
	deck           [52]string
	drawable       []bool
	playerCards    []int
	dealerCards    []int
	playerTotal    int
	dealerTotal    int
	turn           string // "player", "dealer", "end"
	message        string
	showDealerHand bool
	cursor         int
	options        []string
}

func initialModel() model {
	deck := [52]string{
		" 2♥", " 2♦", " 2♣", " 2♠",
		" 3♥", " 3♦", " 3♣", " 3♠",
		" 4♥", " 4♦", " 4♣", " 4♠",
		" 5♥", " 5♦", " 5♣", " 5♠",
		" 6♥", " 6♦", " 6♣", " 6♠",
		" 7♥", " 7♦", " 7♣", " 7♠",
		" 8♥", " 8♦", " 8♣", " 8♠",
		" 9♥", " 9♦", " 9♣", " 9♠",
		"10♥", "10♦", "10♣", "10♠",
		" J♥", " J♦", " J♣", " J♠",
		" Q♥", " Q♦", " Q♣", " Q♠",
		" K♥", " K♦", " K♣", " K♠",
		" A♥", " A♦", " A♣", " A♠"}

	drawable := make([]bool, 52)
	for i := range drawable {
		drawable[i] = true
	}

	playerCards := make([]int, 0, 11)
	dealerCards := make([]int, 0, 10)
	for range []int{1, 2} {
		playerCards, drawable = drawOne(playerCards, drawable)
		dealerCards, drawable = drawOne(dealerCards, drawable)
	}

	playerTotal := calculateHand(playerCards)
	dealerTotal := calculateHand(dealerCards)

	message := ""
	turn := "player"
	showDealerHand := false
	options := []string{"Hit", "Stand", "Quit"}

	if playerTotal == 21 || dealerTotal == 21 {
		showDealerHand = true
		turn = "end"
		switch {
		case playerTotal == 21 && dealerTotal != 21:
			message = "Natural Blackjack! Player Wins!"
		case dealerTotal == 21 && playerTotal != 21:
			message = "Natural Blackjack! Dealer Wins!"
		case dealerTotal == 21 && playerTotal == 21:
			message = "Natural Blackjack! It's a Draw!"
		}
		options = []string{"Restart", "Quit"}
	}

	return model{
		deck:           deck,
		drawable:       drawable,
		playerCards:    playerCards,
		dealerCards:    dealerCards,
		playerTotal:    playerTotal,
		dealerTotal:    dealerTotal,
		turn:           turn,
		message:        message,
		showDealerHand: showDealerHand,
		cursor:         0,
		options:        options,
	}
}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.String() {
		case "up":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down":
			if m.cursor < len(m.options)-1 {
				m.cursor++
			}
		case "enter":
			selected := m.options[m.cursor]

			switch selected {
			case "Hit":
				if m.turn != "player" {
					return m, nil
				}
				m.playerCards, m.drawable = drawOne(m.playerCards, m.drawable)
				m.playerTotal = calculateHand(m.playerCards)
				if m.playerTotal > 21 {
					m.message = fmt.Sprintf("Player Busts with %d! Dealer Wins.", m.playerTotal)
					m.turn = "end"
					m.showDealerHand = true
					m.options = []string{"Restart", "Quit"}
					m.cursor = 0
				}

			case "Stand":
				if m.turn != "player" {
					return m, nil
				}
				m.turn = "dealer"
				m.showDealerHand = true

				for m.dealerTotal < 17 {
					m.dealerCards, m.drawable = drawOne(m.dealerCards, m.drawable)
					m.dealerTotal = calculateHand(m.dealerCards)
				}

				if m.dealerTotal > 21 {
					m.message = fmt.Sprintf("Dealer Busts with %d! Player Wins.", m.dealerTotal)
				} else if m.playerTotal > m.dealerTotal {
					m.message = fmt.Sprintf("Player Wins! %d > %d", m.playerTotal, m.dealerTotal)
				} else if m.playerTotal < m.dealerTotal {
					m.message = fmt.Sprintf("Dealer Wins! %d < %d", m.playerTotal, m.dealerTotal)
				} else {
					m.message = fmt.Sprintf("Draw! %d = %d", m.playerTotal, m.dealerTotal)
				}

				m.turn = "end"
				m.options = []string{"Restart", "Quit"}
				m.cursor = 0

			case "Restart":
				return initialModel(), nil

			case "Quit":
				return m, tea.Quit
			}
		}
	}
	return m, nil
}

func (m model) View() string {
	s := "> Blackjack <\n\n"

	s += "Player Cards: "
	for _, c := range m.playerCards {
		s += m.deck[c] + " "
	}
	s += "\n"

	s += "Dealer Cards: "
	if m.showDealerHand {
		for _, c := range m.dealerCards {
			s += m.deck[c] + " "
		}
		s += "\n"
	} else {
		s += m.deck[m.dealerCards[0]] + " ??\n"
	}

	if m.turn == "player" {
		s += "\nUse ↑/↓ to select an option and press Enter to confirm:\n"
	} else if m.turn == "end" {
		s += "\n" + m.message + "\n"
		s += "Use ↑/↓ to select an option and press Enter to confirm:\n"
	}

	for i, option := range m.options {
		cursor := " "
		if i == m.cursor {
			cursor = "→"
		}
		s += fmt.Sprintf("%s %s\n", cursor, option)
	}

	return s
}

// ----------------- Supporting functions ----------------------

func drawOne(hand []int, poolOfCards []bool) ([]int, []bool) {
	card := rand.Intn(52)
	for poolOfCards[card] == false {
		card = rand.Intn(52)
	}
	poolOfCards[card] = false
	hand = append(hand, card)
	return hand, poolOfCards
}

func calculateHand(cards []int) int {
	sum := 0
	ace := 0
	for _, card := range cards {
		switch {
		case card >= 48:
			sum += 11
			ace++
		case card >= 32:
			sum += 10
		case card >= 28:
			sum += 9
		case card >= 24:
			sum += 8
		case card >= 20:
			sum += 7
		case card >= 16:
			sum += 6
		case card >= 12:
			sum += 5
		case card >= 8:
			sum += 4
		case card >= 4:
			sum += 3
		default:
			sum += 2
		}
	}
	for ace > 0 && sum > 21 {
		sum -= 10
		ace--
	}
	return sum
}

func main() {
	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}
}
