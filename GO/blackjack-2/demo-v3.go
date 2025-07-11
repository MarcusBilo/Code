package main

import (
	"fmt"
	tea "github.com/charmbracelet/bubbletea"
	"log"
	"math/rand"
)

// ------------------------ Main ---------------------------------------------------------------------

func main() {

	p := tea.NewProgram(initialModel())
	_, err := p.Run()
	if err != nil {
		log.Fatal(err)
	}
}

// ------------------- Global Constants --------------------------------------------------------------

const (
	turnPlayer = "player"
	turnDealer = "dealer"
	turnEnd    = "end"
)

// ----------------- Bubble Tea Program --------------------------------------------------------------

type model struct {
	deck           [52]string
	drawStack      []int
	playerCards    []int
	dealerCards    []int
	playerTotal    int
	dealerTotal    int
	showDealerHand bool
	turn           string
	message        string
	cursor         int
}

func initialModel() model {

	deck := [52]string{
		"  2♥", "  2♦", "  2♣", "  2♠", // 0-3
		"  3♥", "  3♦", "  3♣", "  3♠",
		"  4♥", "  4♦", "  4♣", "  4♠",
		"  5♥", "  5♦", "  5♣", "  5♠",
		"  6♥", "  6♦", "  6♣", "  6♠",
		"  7♥", "  7♦", "  7♣", "  7♠",
		"  8♥", "  8♦", "  8♣", "  8♠", // ...
		"  9♥", "  9♦", "  9♣", "  9♠",
		" 10♥", " 10♦", " 10♣", " 10♠",
		"  J♥", "  J♦", "  J♣", "  J♠",
		"  Q♥", "  Q♦", "  Q♣", "  Q♠",
		"  K♥", "  K♦", "  K♣", "  K♠",
		"  A♥", "  A♦", "  A♣", "  A♠"} // 48-51

	drawStack := rand.Perm(52)

	playerCards := make([]int, 0, 11) // 11 is the most the player could need: (A A A A 2 2 2 2 3 3 3)
	dealerCards := make([]int, 0, 10) // 10 is the most the dealer could need: (2 2 2 2 3 A A A A 6)
	playerCards, drawStack = drawOneFromStack(playerCards, drawStack)
	dealerCards, drawStack = drawOneFromStack(dealerCards, drawStack)
	playerCards, drawStack = drawOneFromStack(playerCards, drawStack)
	dealerCards, drawStack = drawOneFromStack(dealerCards, drawStack)

	playerTotal, _ := calculateHand(playerCards)
	dealerTotal, _ := calculateHand(dealerCards)

	showDealerHand := false
	turn := turnPlayer
	message := ""

	if playerTotal == 21 || dealerTotal == 21 {
		showDealerHand = true
		turn = turnEnd
		switch {
		case playerTotal == 21 && dealerTotal != 21:
			message = "Natural Blackjack! Player Wins!"
		case playerTotal != 21 && dealerTotal == 21:
			message = "Natural Blackjack! Dealer Wins!"
		case playerTotal == 21 && dealerTotal == 21:
			message = "Natural Blackjack! It's a Draw!"
		}
	}

	return model{
		deck:           deck,
		drawStack:      drawStack,
		playerCards:    playerCards,
		dealerCards:    dealerCards,
		playerTotal:    playerTotal,
		dealerTotal:    dealerTotal,
		showDealerHand: showDealerHand,
		turn:           turn,
		message:        message,
		cursor:         0,
	}
}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	options := m.currentOptions()
	key := keyMsg.String()
	switch key {
	case "up":
		if m.cursor > 0 {
			m.cursor--
		}
	case "down":
		if m.cursor < len(options)-1 {
			m.cursor++
		}
	case "enter":
		selected := options[m.cursor]
		return handleSelection(m, selected)
	}

	return m, nil
}

func (m model) currentOptions() []string {
	if m.turn != turnEnd {
		return []string{"Hit", "Stand", "Quit"}
	} else {
		return []string{"Restart", "Quit"}
	}
}

func handleSelection(m model, selected string) (tea.Model, tea.Cmd) {
	switch selected {

	case "Hit":
		if m.turn != turnPlayer {
			return m, nil
		}
		m.playerCards, m.drawStack = drawOneFromStack(m.playerCards, m.drawStack)
		m.playerTotal, _ = calculateHand(m.playerCards)
		if m.playerTotal > 21 {
			m.message = fmt.Sprintf("Player Busts with %d! Dealer Wins.", m.playerTotal)
			m.showDealerHand = true
			m.turn = turnEnd
			m.cursor = 0
		}
		return m, nil

	case "Stand":
		if m.turn != turnPlayer {
			return m, nil
		}
		m.showDealerHand = true
		m.turn = turnDealer

		// Dealer Hits Soft 17: 0.18% Casino Edge
		var isSoft17 bool
		for {
			m.dealerTotal, isSoft17 = calculateHand(m.dealerCards)
			if m.dealerTotal > 17 || (m.dealerTotal == 17 && !isSoft17) {
				break
			}
			m.dealerCards, m.drawStack = drawOneFromStack(m.dealerCards, m.drawStack)
		}

		/*
			// Dealer Stands on Soft 17: 0.02% Casino Edge
			for m.dealerTotal < 17 {
				m.dealerCards, m.drawStack = drawOneFromStack(m.dealerCards, m.drawStack)
				m.dealerTotal, _ = calculateHand(m.dealerCards)
			}
		*/

		switch {
		case m.dealerTotal > 21:
			m.message = fmt.Sprintf("Dealer Busts with %d! Player Wins.", m.dealerTotal)
		case m.playerTotal > m.dealerTotal:
			m.message = fmt.Sprintf("Player Wins! %d > %d", m.playerTotal, m.dealerTotal)
		case m.playerTotal < m.dealerTotal:
			m.message = fmt.Sprintf("Dealer Wins! %d < %d", m.playerTotal, m.dealerTotal)
		case m.playerTotal == m.dealerTotal:
			m.message = fmt.Sprintf("Draw! %d = %d", m.playerTotal, m.dealerTotal)
		}

		m.turn = turnEnd
		m.cursor = 0
		return m, nil

	case "Restart":
		return initialModel(), nil

	case "Quit":
		return m, tea.Quit
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
		s += m.deck[m.dealerCards[0]] + "   ??\n"
	}

	if m.turn == turnPlayer {
		s += "\nUse ↑/↓ to select and press Enter to confirm:\n"
	} else if m.turn == turnEnd {
		s += "\n" + m.message + "\n"
		s += "Use ↑/↓ to select and press Enter to confirm:\n"
	}

	options := m.currentOptions()
	for i, option := range options {
		cursor := " "
		if i == m.cursor {
			cursor = "→"
		}
		s += fmt.Sprintf("%s %s\n", cursor, option)
	}

	return s
}

// ---------------- Supporting functions -------------------------------------------------------------

func drawOneFromStack(hand []int, drawStack []int) ([]int, []int) {
	if len(drawStack) == 0 {
		log.Fatal("No more cards in deck!")
	}
	drawnCard := drawStack[0]
	newDrawStack := drawStack[1:]
	newHand := append(hand, drawnCard)
	return newHand, newDrawStack
}

func calculateHand(cards []int) (handTotal int, isSoft17 bool) {
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
	if sum == 17 && ace > 0 {
		return sum, true
	}
	return sum, false
}
